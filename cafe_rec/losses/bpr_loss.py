# L_rec: Bayesian Personalized Ranking loss for implicit-feedback recommendation.
from __future__ import annotations

import torch
import torch.nn.functional as F


def bpr_loss(
    pos_score: torch.Tensor,
    neg_score: torch.Tensor,
) -> torch.Tensor:
    """BPR loss: -E[ log sigmoid(pos_score - neg_score) ].

    Expects the caller to have already scored the positive and a randomly
    sampled negative item through the ConstrainedScorer. In CaFE-Rec those
    are `model_output["rec_score"]` and `model_output["neg_rec_score"]`,
    both populated by `CaFeRec.forward(..., neg_item_ids=...)`.

    Args:
        pos_score: [B] or [B, 1] score for each (user, positive_item).
        neg_score: [B] or [B, 1] score for each (user, negative_item).

    Returns:
        Scalar loss tensor.
    """
    pos_score = pos_score.view(-1)
    neg_score = neg_score.view(-1)
    # logsigmoid is numerically safer than log(sigmoid(...)).
    return -F.logsigmoid(pos_score - neg_score).mean()
