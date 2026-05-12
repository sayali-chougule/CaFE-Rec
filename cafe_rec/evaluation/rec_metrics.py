# Recommendation metrics: Recall@K, NDCG@K, HitRate@K.
from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ArrayLike = Union[np.ndarray, torch.Tensor, List[float]]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# 1. Recall@K
# ---------------------------------------------------------------------------
def recall_at_k(
    scores: ArrayLike,
    ground_truth_item: int,
    k: int,
) -> float:
    """Recall@K for a single ranking.

    Formula:
        Recall@K = 1  if  ground_truth_item in top-K by score  else  0

    Args:
        scores: shape [num_candidates] — higher is better.
        ground_truth_item: index of the positive candidate.
        k: cutoff.

    Returns:
        1.0 or 0.0 (float).
    """
    s = _to_numpy(scores)
    # Rank candidates descending by score; `argsort(-s)` puts best first.
    top_k = np.argsort(-s, kind="stable")[:k]
    return float(int(ground_truth_item) in set(int(x) for x in top_k))


# ---------------------------------------------------------------------------
# 2. NDCG@K
# ---------------------------------------------------------------------------
def ndcg_at_k(
    scores: ArrayLike,
    ground_truth_item: int,
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain @ K (single relevant item).

    For a single relevant item with binary relevance:
        DCG@K  = 1 / log2(rank + 2)   if rank < K   else 0
        IDCG@K = 1 / log2(0 + 2) = 1  (ideal: relevant at position 1)
        NDCG@K = DCG / IDCG = DCG

    `rank` is the 0-indexed position of the ground-truth item in the
    descending score order.
    """
    s = _to_numpy(scores)
    order = np.argsort(-s, kind="stable")
    rank_arr = np.where(order == int(ground_truth_item))[0]
    if rank_arr.size == 0:
        return 0.0
    rank = int(rank_arr[0])
    if rank >= int(k):
        return 0.0
    return float(1.0 / np.log2(rank + 2))


# ---------------------------------------------------------------------------
# 3. Full eval loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_recommendations(
    model,
    eval_dataset,
    config,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Dict[str, float]:
    """Recall@K / NDCG@K over an EvaluationDataset.

    The eval dataset yields (user, pos_item, neg_items[100]); we treat
    position 0 of the concatenated candidate list as the ground truth
    and score all candidates with the frozen-graph CaFE-Rec pipeline
    (hard aspect selection).

    Returns a flat dict like
        {"Recall@5": ..., "Recall@10": ..., "NDCG@5": ..., ...}
    averaged over the dataset.
    """
    from data.dataset import evaluation_collate_fn

    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=evaluation_collate_fn,
    )

    # Cache embeddings: edge_index doesn't change during eval.
    all_user_emb, all_item_emb = model.get_embeddings()

    ks = list(config.top_k_list)
    acc = {f"Recall@{k}": [] for k in ks}
    acc.update({f"NDCG@{k}": [] for k in ks})

    for users, pos_items, neg_items in tqdm(loader, desc="eval rec"):
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        B, Nneg = neg_items.shape

        # Candidates: [pos | negs] -> ground truth is index 0.
        cands = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
        ncand = cands.size(1)

        user_emb = all_user_emb[users]
        u_exp = (
            user_emb.unsqueeze(1).expand(-1, ncand, -1).reshape(B * ncand, -1)
        )
        i_exp = all_item_emb[cands.reshape(-1)]

        selected, _ = model.aspect_selector(u_exp, i_exp, hard=True)
        scores = model.constrained_scorer(u_exp, i_exp, selected).view(B, ncand)

        scores_np = scores.cpu().numpy()
        for i in range(B):
            for k in ks:
                acc[f"Recall@{k}"].append(recall_at_k(scores_np[i], 0, k))
                acc[f"NDCG@{k}"].append(ndcg_at_k(scores_np[i], 0, k))

    return {k: float(np.mean(v)) for k, v in acc.items()}
