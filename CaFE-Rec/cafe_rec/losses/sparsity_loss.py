# L_sparse: L1 penalty on aspect selection logits to keep the chosen set small (3-5).
from __future__ import annotations

import torch


def sparsity_loss(aspect_logits: torch.Tensor) -> torch.Tensor:
    """Mean |logits| over (batch, aspects).

    Encourages the AspectSelector to pick few, decisive aspects. The
    external scalar weight (`config.lambda_sparse`) is applied by the
    training loop, not here -- that way the training log shows the raw
    unweighted signal, which is easier to compare across runs.

    Args:
        aspect_logits: [B, A] raw (pre-Gumbel) selector logits.

    Returns:
        Scalar L1 mean.
    """
    return aspect_logits.abs().mean()
