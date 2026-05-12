# 2-layer MLP + Gumbel-Softmax that selects 3-5 aspects from the aspect vocabulary.
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AspectSelector(nn.Module):
    """Stochastic aspect gating — the CaFE-Rec information bottleneck.

    A 2-layer MLP produces per-aspect logits from concatenated user/item
    embeddings. Those logits are turned into *independent* soft/hard binary
    selections via Gumbel-Sigmoid (implemented as 2-way Gumbel-Softmax over
    the pair (select, don't-select)). Because selections are independent
    per aspect, multiple aspects can be active simultaneously; the L1
    sparsity loss (applied externally to `aspect_logits`) pushes the number
    of active aspects toward the configured budget (typically 3-5).
    """

    def __init__(
        self,
        input_dim: int,
        num_aspects: int = 50,
        hidden_dim: int = 256,
        tau_start: float = 1.0,
        tau_end: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_aspects = int(num_aspects)
        self.hidden_dim = int(hidden_dim)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        # Current temperature: mutable, updated via anneal_temperature().
        self.tau = float(tau_start)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.hidden_dim, self.num_aspects),
        )

        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward: produce per-aspect selection weights
    # ------------------------------------------------------------------
    def forward(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select aspects via Gumbel-Sigmoid.

        Args:
            user_emb: [B, D_u]
            item_emb: [B, D_i] (D_u + D_i must equal input_dim)
            hard: if True, straight-through hard 0/1 selection (inference).
                  If False, differentiable soft selection in [0, 1] (training).

        Returns:
            selected_aspects: [B, num_aspects] — soft or hard weights.
            aspect_logits:    [B, num_aspects] — raw logits (pre-Gumbel),
                              suitable for the L1 sparsity loss.
        """
        x = torch.cat([user_emb, item_emb], dim=-1)
        logits = self.mlp(x)

        # Pair each aspect logit with a zero "off" logit to form a 2-class
        # categorical; Gumbel-Softmax over that pair is a Gumbel-Sigmoid
        # sample for the aspect, independent across aspects.
        pair = torch.stack([logits, torch.zeros_like(logits)], dim=-1)
        tau = max(float(self.tau), 1e-4)  # guard against zero tau
        gumbel = F.gumbel_softmax(pair, tau=tau, hard=hard, dim=-1)
        selected = gumbel[..., 0]
        return selected, logits

    # ------------------------------------------------------------------
    # Temperature annealing (called at the top of each training epoch)
    # ------------------------------------------------------------------
    def anneal_temperature(
        self, current_epoch: int, total_epochs: int
    ) -> float:
        """Linearly interpolate tau from tau_start to tau_end."""
        if total_epochs <= 1:
            self.tau = self.tau_end
        else:
            frac = min(max(current_epoch / (total_epochs - 1), 0.0), 1.0)
            self.tau = self.tau_start + (self.tau_end - self.tau_start) * frac
        return self.tau

    # ------------------------------------------------------------------
    # Interpretability helper
    # ------------------------------------------------------------------
    @staticmethod
    def get_top_k_aspects(
        aspect_weights: torch.Tensor, k: int = 5
    ) -> torch.Tensor:
        """Return indices of the k highest-weight aspects per row.

        Args:
            aspect_weights: [B, num_aspects] soft or hard weights.
            k: number of aspects to keep.

        Returns:
            LongTensor [B, k] of aspect indices, sorted by descending weight.
        """
        k = min(int(k), aspect_weights.size(-1))
        _, idx = aspect_weights.topk(k, dim=-1)
        return idx

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            "AspectSelector(\n"
            f"  input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_aspects={self.num_aspects},\n"
            f"  tau_start={self.tau_start}, tau_end={self.tau_end}, "
            f"tau_current={self.tau:.4f},\n"
            f"  mlp=Linear({self.input_dim}->{self.hidden_dim}) -> ReLU -> "
            f"Dropout -> Linear({self.hidden_dim}->{self.num_aspects})\n"
            ")"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, D_u, D_i, N = 8, 64, 64, 50
    user_emb = torch.randn(B, D_u, requires_grad=True)
    item_emb = torch.randn(B, D_i, requires_grad=True)

    sel = AspectSelector(
        input_dim=D_u + D_i, num_aspects=N, hidden_dim=256
    )
    print(sel)

    # --- Soft (training) ---
    sel.train()
    soft, logits = sel(user_emb, item_emb, hard=False)
    print(f"\n[soft]  selected={tuple(soft.shape)}  "
          f"logits={tuple(logits.shape)}")
    print(f"        weights range [{soft.min().item():.3f}, "
          f"{soft.max().item():.3f}]  "
          f"mean active (>0.5) per row = "
          f"{(soft > 0.5).float().sum(dim=-1).mean().item():.2f}")
    assert soft.shape == (B, N) and logits.shape == (B, N)
    # Gradient flow
    soft.sum().backward()
    assert user_emb.grad is not None and item_emb.grad is not None

    # --- Hard (inference) ---
    sel.eval()
    with torch.no_grad():
        hard, _ = sel(user_emb, item_emb, hard=True)
    print(f"[hard]  values unique: {sorted(hard.unique().tolist())}  "
          f"mean active per row = "
          f"{hard.sum(dim=-1).mean().item():.2f}")
    assert set(hard.unique().tolist()).issubset({0.0, 1.0})

    # --- Annealing ---
    taus = [sel.anneal_temperature(e, 10) for e in range(10)]
    print(f"\n[anneal] tau over 10 epochs: "
          f"{[round(t, 3) for t in taus]}")
    assert abs(taus[0] - 1.0) < 1e-6 and abs(taus[-1] - 0.1) < 1e-6

    # --- Top-k ---
    topk = AspectSelector.get_top_k_aspects(soft.detach(), k=5)
    print(f"[top-5] indices shape: {tuple(topk.shape)}")
    assert topk.shape == (B, 5)

    print("\nOK: all AspectSelector checks pass.")
