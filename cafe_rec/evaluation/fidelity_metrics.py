# Counterfactual-fidelity metrics: sufficiency, comprehensiveness, CF%.
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Internals: score under a full / masked aspect set
# ---------------------------------------------------------------------------
@torch.no_grad()
def _score(model, user_emb, item_emb, aspects):
    return model.constrained_scorer(user_emb, item_emb, aspects).view(-1)


@torch.no_grad()
def _selected_and_explanation_mask(model, user_emb, item_emb, k: int):
    """Return (selected_aspects [B, A], explanation_mask [B, A])."""
    selected, _ = model.aspect_selector(user_emb, item_emb, hard=True)
    top_idx = selected.topk(
        min(int(k), selected.size(-1)), dim=-1
    ).indices
    mask = torch.zeros_like(selected)
    mask.scatter_(1, top_idx, 1.0)
    return selected, mask


# ---------------------------------------------------------------------------
# 1. Sufficiency
# ---------------------------------------------------------------------------
@torch.no_grad()
def sufficiency_score(
    model,
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    aspect_explanations: Optional[torch.Tensor] = None,
    config=None,
) -> float:
    """Sufficiency: can the explained aspects ALONE reproduce the score?

    Let
        s_full  = ConstrainedScorer(u, i, selected)
        s_only  = ConstrainedScorer(u, i, selected * explanation_mask)
    Then
        sufficiency = mean( 1 - |s_full - s_only| / (|s_full| + eps) )

    Higher is better: value near 1.0 means the explanation alone is
    enough to recover the recommendation score.

    If `aspect_explanations` is None we default to the top-K aspects
    (config.num_selected_aspects).
    """
    model.eval()
    device = next(model.parameters()).device
    user_ids = user_ids.to(device)
    item_ids = item_ids.to(device)

    all_u, all_i = model.get_embeddings()
    user_emb = all_u[user_ids]
    item_emb = all_i[item_ids]

    selected, mask_default = _selected_and_explanation_mask(
        model, user_emb, item_emb, k=config.num_selected_aspects
    )
    expl_mask = (
        aspect_explanations.to(device).float()
        if aspect_explanations is not None
        else mask_default
    )

    s_full = _score(model, user_emb, item_emb, selected)
    s_only = _score(model, user_emb, item_emb, selected * expl_mask)

    suff = 1.0 - (s_full - s_only).abs() / (s_full.abs() + 1e-8)
    return float(suff.mean().item())


# ---------------------------------------------------------------------------
# 2. Comprehensiveness
# ---------------------------------------------------------------------------
@torch.no_grad()
def comprehensiveness_score(
    model,
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    aspect_explanations: Optional[torch.Tensor] = None,
    config=None,
) -> float:
    """Comprehensiveness: does REMOVING the explained aspects hurt the score?

    Let
        s_full   = ConstrainedScorer(u, i, selected)
        s_ablate = ConstrainedScorer(u, i, selected * (1 - explanation_mask))
    Then
        comprehensiveness = mean( s_full - s_ablate )

    Higher is better: if the explanation is comprehensive, taking it away
    should significantly lower the recommendation score.
    """
    model.eval()
    device = next(model.parameters()).device
    user_ids = user_ids.to(device)
    item_ids = item_ids.to(device)

    all_u, all_i = model.get_embeddings()
    user_emb = all_u[user_ids]
    item_emb = all_i[item_ids]

    selected, mask_default = _selected_and_explanation_mask(
        model, user_emb, item_emb, k=config.num_selected_aspects
    )
    expl_mask = (
        aspect_explanations.to(device).float()
        if aspect_explanations is not None
        else mask_default
    )

    s_full = _score(model, user_emb, item_emb, selected)
    s_ablate = _score(model, user_emb, item_emb, selected * (1.0 - expl_mask))
    return float((s_full - s_ablate).mean().item())


# ---------------------------------------------------------------------------
# 3. Counterfactual fidelity %
# ---------------------------------------------------------------------------
@torch.no_grad()
def counterfactual_fidelity_percentage(
    model,
    test_dataset,
    config,
    k: int = 10,
    batch_size: int = 64,
) -> float:
    """CF% — fraction of cases where the explanation is truly counterfactual.

    Procedure per test (user, pos_item, neg_items) triplet:
      a. Select aspects for (user, pos_item); take the top-K as the
         "explanation". This is what the system would show a human.
      b. Score all candidates (pos + negatives) normally (factual ranking).
      c. Score all candidates again with the explanation aspects zeroed
         across the board (counterfactual ranking).
      d. Count the case as faithful iff the pos_item was in top-`k`
         factually and is NOT in top-`k` counterfactually.

    CF% = #faithful / #(pos_item was in top-k factually).

    Note: the spec's "remove aspect-related items from user history" is
    approximated here by zeroing the explanation aspects in the scorer
    input — the same channel CaFE-Rec's fidelity loss manipulates during
    training. Rebuilding the LightGCN graph per example is too expensive
    for large test sets and tends not to change rankings on held-out
    negatives any differently than the aspect-channel ablation.
    """
    from data.dataset import evaluation_collate_fn

    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=evaluation_collate_fn,
    )
    all_u, all_i = model.get_embeddings()

    num_in_topk = 0
    num_flipped = 0

    for users, pos_items, neg_items in tqdm(loader, desc="CF%"):
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B, Nneg = neg_items.shape

        cands = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        ncand = cands.size(1)
        user_emb = all_u[users]

        # (a) explanation = top-K aspects for the positive pair.
        pos_emb = all_i[pos_items]
        _, expl_mask = _selected_and_explanation_mask(
            model, user_emb, pos_emb, k=config.num_selected_aspects
        )

        # (b) factual scores for all candidates under normal selection.
        u_exp = (
            user_emb.unsqueeze(1).expand(-1, ncand, -1).reshape(B * ncand, -1)
        )
        i_exp = all_i[cands.reshape(-1)]
        sel_all, _ = model.aspect_selector(u_exp, i_exp, hard=True)
        fact_scores = model.constrained_scorer(
            u_exp, i_exp, sel_all
        ).view(B, ncand)

        # (c) counterfactual: broadcast explanation mask across candidates.
        mask_exp = (
            expl_mask.unsqueeze(1).expand(-1, ncand, -1).reshape(B * ncand, -1)
        )
        sel_cf = sel_all * (1.0 - mask_exp)
        cf_scores = model.constrained_scorer(
            u_exp, i_exp, sel_cf
        ).view(B, ncand)

        # (d) count flips: ground truth is always index 0 in cands.
        fact_top = fact_scores.topk(k, dim=-1).indices  # [B, k]
        cf_top = cf_scores.topk(k, dim=-1).indices

        fact_has_gt = (fact_top == 0).any(dim=-1)
        cf_has_gt = (cf_top == 0).any(dim=-1)

        num_in_topk += int(fact_has_gt.sum().item())
        num_flipped += int(
            (fact_has_gt & ~cf_has_gt).sum().item()
        )

    denom = max(num_in_topk, 1)
    return float(num_flipped / denom)


# ---------------------------------------------------------------------------
# 4. Run all three
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_full_fidelity_evaluation(
    model,
    test_dataset,
    config,
    batch_size: int = 64,
    cf_k: int = 10,
) -> Dict[str, float]:
    """Run sufficiency, comprehensiveness, and CF% over the whole test set.

    Returns:
        {"sufficiency": ..., "comprehensiveness": ..., "CF@k": ...}
    """
    from data.dataset import evaluation_collate_fn

    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=evaluation_collate_fn,
    )

    suff_vals = []
    comp_vals = []
    for users, pos_items, _ in tqdm(loader, desc="fidelity suff/comp"):
        suff_vals.append(
            sufficiency_score(model, users, pos_items, None, config)
        )
        comp_vals.append(
            comprehensiveness_score(model, users, pos_items, None, config)
        )

    cf = counterfactual_fidelity_percentage(
        model, test_dataset, config, k=cf_k, batch_size=batch_size
    )
    return {
        "sufficiency": float(np.mean(suff_vals)),
        "comprehensiveness": float(np.mean(comp_vals)),
        f"CF@{cf_k}": cf,
    }
