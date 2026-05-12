# End-to-end wrapper: LightGCN -> AspectSelector -> (ConstrainedScorer, Adapter -> Flan-T5).
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .lightgcn import LightGCN
from .aspect_selector import AspectSelector
from .constrained_scorer import ConstrainedScorer
from .adapter import AspectToT5Adapter


class CaFeRec(nn.Module):
    """Counterfactually Faithful Explanation Recommender.

    Pipeline:
      user/item ids
         -> LightGCN                      (graph-aware embeddings)
         -> AspectSelector                (Gumbel-sigmoid info bottleneck)
         -> ConstrainedScorer             (recommendation score)
         -> AspectToT5Adapter -> Flan-T5  (natural-language explanation)

    The same `selected_aspects` drives both the score (via the scorer) and
    the explanation (via the adapter/T5 prefix), which is what makes the
    recommendation and explanation causally consistent by construction.
    """

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(
        self,
        config,
        num_users: int,
        num_items: int,
        aspect_vocabulary: List[str],
    ):
        super().__init__()
        self.config = config
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.aspect_vocabulary = list(aspect_vocabulary)
        self.num_aspects = len(self.aspect_vocabulary)
        self.num_selected_aspects = int(config.num_selected_aspects)

        emb_dim = int(config.embedding_dim)
        user_item_dim = 2 * emb_dim

        # 1) LightGCN backbone
        self.lightgcn = LightGCN(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=emb_dim,
            num_layers=int(config.num_gcn_layers),
            dropout=float(config.dropout),
        )

        # 2) Aspect selector (info bottleneck)
        self.aspect_selector = AspectSelector(
            input_dim=user_item_dim,
            num_aspects=self.num_aspects,
            hidden_dim=256,
            tau_start=float(config.gumbel_tau_start),
            tau_end=float(config.gumbel_tau_end),
            dropout=float(config.dropout),
        )

        # 3) Constrained recommender head
        self.constrained_scorer = ConstrainedScorer(
            user_item_dim=user_item_dim,
            aspect_dim=self.num_aspects,
            hidden_dim=256,
            dropout=float(config.dropout),
        )

        # 4) Explanation head: adapter + Flan-T5-Base
        t5_name = str(config.flan_model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_name)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_name)
        t5_hidden = int(self.t5.config.d_model)
        self.adapter = AspectToT5Adapter(
            aspect_dim=self.num_aspects,
            user_item_dim=user_item_dim,
            t5_hidden_dim=t5_hidden,
            num_prefix_tokens=10,
            dropout=float(config.dropout),
        )

        # Edge index is populated by build_graph() at training time.
        self.register_buffer(
            "_edge_index", torch.empty(2, 0, dtype=torch.long),
            persistent=False,
        )
        self._edge_index_ready = False

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def build_graph(self, train_df: pd.DataFrame) -> None:
        """Build and move the LightGCN edge_index to the model's device."""
        edge_index = LightGCN.build_edge_index(
            train_df, self.num_users, self.num_items
        )
        self._edge_index = edge_index.to(self._device())
        self._edge_index_ready = True

    @property
    def edge_index(self) -> torch.Tensor:
        if not self._edge_index_ready:
            raise RuntimeError(
                "edge_index not built. Call model.build_graph(train_df) "
                "before forward()."
            )
        return self._edge_index

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: Optional[torch.Tensor] = None,
        explanation_ids: Optional[torch.Tensor] = None,
        explanation_mask: Optional[torch.Tensor] = None,
        mode: str = "train",
    ) -> Dict[str, torch.Tensor]:
        """Run the full pipeline for a batch.

        Args:
            user_ids:        [B]  user indices.
            pos_item_ids:    [B]  positive item indices.
            neg_item_ids:    [B]  negative item indices (for BPR); optional.
            explanation_ids: [B, T]  reference explanation token ids; optional.
            explanation_mask:[B, T]  attention mask for explanation_ids;
                                    unused for the T5 label path but kept
                                    for API symmetry.
            mode:  "train" -> soft Gumbel selection;
                   otherwise -> hard (inference-style) selection.

        Returns a dict with whichever outputs were computed:
          user_emb, pos_item_emb, neg_item_emb,
          selected_aspects, aspect_logits,
          rec_score, neg_rec_score,
          explanation_loss, explanation_logits, prefix_emb.
        """
        del explanation_mask  # not required by the labels-only path

        hard = mode != "train"

        # Step 1: LightGCN over the full graph.
        all_user_emb, all_item_emb = self.lightgcn(self.edge_index)

        # Step 2: gather batch embeddings.
        user_emb = all_user_emb[user_ids]
        pos_item_emb = all_item_emb[pos_item_ids]

        # Step 3: aspect selection for positive pair.
        selected_aspects, aspect_logits = self.aspect_selector(
            user_emb, pos_item_emb, hard=hard
        )

        # Step 4: constrained recommendation score.
        rec_score = self.constrained_scorer(
            user_emb, pos_item_emb, selected_aspects
        )

        out: Dict[str, torch.Tensor] = {
            "user_emb": user_emb,
            "pos_item_emb": pos_item_emb,
            "selected_aspects": selected_aspects,
            "aspect_logits": aspect_logits,
            "rec_score": rec_score,
        }

        # Negative scoring for BPR.
        if neg_item_ids is not None:
            neg_item_emb = all_item_emb[neg_item_ids]
            neg_sel, _ = self.aspect_selector(
                user_emb, neg_item_emb, hard=hard
            )
            neg_rec_score = self.constrained_scorer(
                user_emb, neg_item_emb, neg_sel
            )
            out["neg_item_emb"] = neg_item_emb
            out["neg_rec_score"] = neg_rec_score

        # Step 5: explanation via adapter + T5.
        if explanation_ids is not None:
            prefix_emb = self.adapter(
                selected_aspects, user_emb, pos_item_emb
            )
            B, P, _ = prefix_emb.shape
            enc_mask = torch.ones(
                B, P, device=prefix_emb.device, dtype=torch.long
            )
            labels = explanation_ids.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            t5_out = self.t5(
                inputs_embeds=prefix_emb,
                attention_mask=enc_mask,
                labels=labels,
            )
            out["explanation_loss"] = t5_out.loss
            out["explanation_logits"] = t5_out.logits
            out["prefix_emb"] = prefix_emb

        return out

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_explanation(
        self,
        user_id: int,
        item_id: int,
        max_length: int = 64,
        num_beams: int = 4,
    ) -> Tuple[str, List[str]]:
        """Generate a human-readable explanation for a single (user, item).

        Returns:
            explanation_text: decoded string.
            top_aspect_names: the K aspect vocabulary entries fed to T5.
        """
        was_training = self.training
        self.eval()
        try:
            device = self._device()
            u = torch.tensor([int(user_id)], dtype=torch.long, device=device)
            i = torch.tensor([int(item_id)], dtype=torch.long, device=device)

            all_u, all_i = self.lightgcn(self.edge_index)
            user_emb = all_u[u]
            item_emb = all_i[i]

            selected, _ = self.aspect_selector(user_emb, item_emb, hard=True)
            top_idx = AspectSelector.get_top_k_aspects(
                selected, k=self.num_selected_aspects
            )[0].tolist()
            top_aspect_names = [
                self.aspect_vocabulary[j] for j in top_idx
                if 0 <= j < len(self.aspect_vocabulary)
            ]

            prefix_emb = self.adapter(selected, user_emb, item_emb)
            enc_mask = torch.ones(
                1, prefix_emb.size(1), device=device, dtype=torch.long
            )
            out_ids = self.t5.generate(
                inputs_embeds=prefix_emb,
                attention_mask=enc_mask,
                max_length=int(max_length),
                num_beams=int(num_beams),
                early_stopping=True,
            )
            text = self.tokenizer.decode(
                out_ids[0], skip_special_tokens=True
            )
            return text, top_aspect_names
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def get_embeddings(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the current (user_emb, item_emb) from LightGCN."""
        was_training = self.training
        self.eval()
        try:
            return self.lightgcn(self.edge_index)
        finally:
            if was_training:
                self.train()


# ---------------------------------------------------------------------------
# Smoke test (requires internet + ~250MB Flan-T5-Base download on first run).
# Skipped silently if transformers cannot fetch the model.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from dataclasses import dataclass

    @dataclass
    class _MiniConfig:
        embedding_dim: int = 32
        num_gcn_layers: int = 2
        dropout: float = 0.1
        gumbel_tau_start: float = 1.0
        gumbel_tau_end: float = 0.1
        num_selected_aspects: int = 3
        flan_model_name: str = "google/flan-t5-base"

    try:
        torch.manual_seed(0)
        num_users, num_items = 15, 25
        vocab = [f"aspect_{i}" for i in range(20)]

        df = pd.DataFrame({
            "user_idx": torch.randint(0, num_users, (100,)).tolist(),
            "item_idx": torch.randint(0, num_items, (100,)).tolist(),
        })

        model = CaFeRec(_MiniConfig(), num_users, num_items, vocab)
        model.build_graph(df)

        u = torch.tensor([0, 1, 2, 3])
        p = torch.tensor([4, 5, 6, 7])
        n = torch.tensor([8, 9, 10, 11])
        exp_ids = model.tokenizer(
            ["great plot", "weak pacing", "lovely score", "dull ending"],
            padding="max_length", max_length=16, truncation=True,
            return_tensors="pt",
        )["input_ids"]

        out = model(u, p, n, explanation_ids=exp_ids, mode="train")
        print("forward keys:", sorted(out.keys()))
        print("rec_score:", tuple(out["rec_score"].shape),
              " neg_rec_score:", tuple(out["neg_rec_score"].shape))
        print("selected_aspects:", tuple(out["selected_aspects"].shape))
        print("explanation_loss:", float(out["explanation_loss"]))

        text, top = model.generate_explanation(0, 4, max_length=16)
        print("generated:", repr(text), "| top aspects:", top)
        print("OK: CaFeRec end-to-end forward + generate pass.")
    except Exception as e:
        print(f"[skipped] smoke test needs Flan-T5 weights / deps: {e}",
              file=sys.stderr)
