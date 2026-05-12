# L_faith (NOVEL): counterfactual fidelity. Mask the selected aspects and penalize when
# the recommendation does NOT change, forcing selected aspects to be genuinely causal.
from __future__ import annotations

from typing import Tuple

import torch


def counterfactual_fidelity_loss(
    constrained_scorer,
    user_emb: torch.Tensor,
    pos_item_emb: torch.Tensor,
    selected_aspects: torch.Tensor,
    margin: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Counterfactual fidelity loss (CaFE-Rec's novel contribution).

    --------------------------------------------------------------------
    Why this loss exists
    --------------------------------------------------------------------
    A self-explaining recommender can still be dishonest: the model could
    pick aspects that LOOK like a plausible explanation while its actual
    decision is driven by some other signal (user id priors, popularity,
    etc.). If that is the case, removing the "explaining" aspects will
    leave the score unchanged -- the aspects are correlative, not causal.

    Counterfactual fidelity asks a direct causal question:
        "If I take away the aspects the model just said were its reasons,
         does the recommendation actually change?"

    If yes -> the aspects really did drive the decision (faithful).
    If no  -> the aspects were a post-hoc rationalization (unfaithful).

    The loss penalizes the second case and so pushes training toward
    explanations that are causally grounded in the scorer.

    --------------------------------------------------------------------
    Steps
    --------------------------------------------------------------------
    1. Normal (factual) score — what the model recommends with the
       aspects it chose.

    2. Build the counterfactual: zero out ALL selected aspects. Because
       the ConstrainedScorer's aspect input is the ONLY channel by which
       aspect information reaches the score, zeroing it simulates
       "having explained nothing".

    3. Counterfactual score — what the model recommends when the stated
       reasons have been removed.

    4. Score drop = |factual - counterfactual|. A large drop means the
       aspects mattered; a small drop means they did not.

    5. Margin hinge: we don't just want any drop, we want at least
       `margin` worth. If the drop is >= margin, the loss is 0; otherwise
       the shortfall is penalized linearly. Without the margin the loss
       would have a trivial solution at drop = epsilon.

    Args:
        constrained_scorer: the ConstrainedScorer module (so we can
            re-score under the counterfactual without touching any
            other CaFE-Rec component).
        user_emb: [B, D_u] from LightGCN.
        pos_item_emb: [B, D_i] from LightGCN.
        selected_aspects: [B, A] from AspectSelector (soft or hard).
        margin: minimum score drop we require; below this we penalize.

    Returns:
        fidelity_loss: scalar — the training signal.
        mean_drop:     scalar — the average |factual - counterfactual|,
                       useful to log and watch climb toward the margin.
    """
    # (1) Factual score: the recommendation the model actually makes.
    score_normal = constrained_scorer(
        user_emb, pos_item_emb, selected_aspects
    ).view(-1)

    # (2) Build the counterfactual aspect vector -- every selected aspect
    # is zeroed, so the scorer loses all aspect-channel information.
    aspects_masked = torch.zeros_like(selected_aspects)

    # (3) Counterfactual score under "no aspects".
    score_masked = constrained_scorer(
        user_emb, pos_item_emb, aspects_masked
    ).view(-1)

    # (4) How much did the score move when we took the reasons away?
    # Absolute value: we care about magnitude of causal effect, not sign.
    score_drop = torch.abs(score_normal - score_masked)

    # (5) Hinge at `margin`. If the drop already exceeds the margin the
    # example contributes zero loss; otherwise we push it toward the margin.
    fidelity_loss = torch.clamp(margin - score_drop, min=0.0).mean()

    return fidelity_loss, score_drop.mean().detach()
