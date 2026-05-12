# L_text: token-level cross-entropy between generated and reference explanation text.
from __future__ import annotations

import torch
import torch.nn as nn


def text_loss(
    explanation_logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Standard seq2seq cross-entropy for T5 decoder outputs.

    Args:
        explanation_logits: [B, T, V] raw logits from T5's LM head.
        target_ids: [B, T] gold token ids. Pad positions should already be
            set to `ignore_index` by the training loop (T5 convention: -100).
        ignore_index: label value to skip in the loss (default -100).

    Returns:
        Scalar mean cross-entropy over non-ignored tokens.
    """
    B, T, V = explanation_logits.shape
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    return loss_fn(
        explanation_logits.reshape(B * T, V),
        target_ids.reshape(B * T),
    )
