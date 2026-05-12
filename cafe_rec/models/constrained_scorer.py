# MLP scoring using ONLY the selected aspects (information bottleneck).
from __future__ import annotations

import torch
import torch.nn as nn


class ConstrainedScorer(nn.Module):
    """Recommendation head that sees ONLY the selected aspects.

    The scorer receives the concatenation of (user_emb, item_emb,
    selected_aspects). Because `selected_aspects` is produced by the
    AspectSelector's Gumbel gate, any aspect that is *not* selected is
    multiplied by ~0 and therefore cannot influence the score -- the
    information bottleneck that enforces consistency between what we
    recommend and what we explain.

    `forward_masked` additionally zeros a caller-supplied subset of
    aspects, enabling the counterfactual fidelity loss: if masking the
    selected aspects does NOT change the score, those aspects were not
    causal, and the loss penalizes that.
    """

    def __init__(
        self,
        user_item_dim: int,
        aspect_dim: int = 50,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.user_item_dim = int(user_item_dim)
        self.aspect_dim = int(aspect_dim)
        self.hidden_dim = int(hidden_dim)
        in_dim = self.user_item_dim + self.aspect_dim
        h2 = self.hidden_dim // 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(self.hidden_dim, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(h2, 1),
            nn.Sigmoid(),
        )

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def _score(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        aspects: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([user_emb, item_emb, aspects], dim=-1)
        return self.net(x)

    def forward(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        selected_aspects: torch.Tensor,
    ) -> torch.Tensor:
        """Score a batch using the aspects picked by AspectSelector.

        Args:
            user_emb: [B, D_u]
            item_emb: [B, D_i]   (D_u + D_i must equal user_item_dim)
            selected_aspects: [B, aspect_dim]
        Returns:
            [B, 1] sigmoid score.
        """
        return self._score(user_emb, item_emb, selected_aspects)

    def forward_masked(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        selected_aspects: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Same as forward, but multiplicatively masks aspects first.

        Args:
            mask: [B, aspect_dim]. `1` keeps an aspect, `0` removes it.
                  Cast to float internally so callers can pass bool or int.
        Returns:
            [B, 1] sigmoid score under the counterfactual aspect set.
        """
        masked = selected_aspects * mask.to(selected_aspects.dtype)
        return self._score(user_emb, item_emb, masked)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, D_u, D_i, N = 8, 64, 64, 50
    user_emb = torch.randn(B, D_u, requires_grad=True)
    item_emb = torch.randn(B, D_i, requires_grad=True)
    selected = torch.rand(B, N)  # soft weights in [0, 1]

    scorer = ConstrainedScorer(user_item_dim=D_u + D_i, aspect_dim=N)
    print(scorer)

    # Forward
    s = scorer(user_emb, item_emb, selected)
    print(f"\n[forward]        score shape={tuple(s.shape)}  "
          f"range=[{s.min().item():.3f}, {s.max().item():.3f}]")
    assert s.shape == (B, 1)
    assert (s >= 0).all() and (s <= 1).all()

    # Masked: drop the top-5 aspects per row and confirm shape/range are OK
    top5 = selected.topk(5, dim=-1).indices
    keep_mask = torch.ones_like(selected)
    keep_mask.scatter_(1, top5, 0.0)  # zero-out the selected aspects
    s_cf = scorer.forward_masked(user_emb, item_emb, selected, keep_mask)
    print(f"[forward_masked] score shape={tuple(s_cf.shape)}  "
          f"range=[{s_cf.min().item():.3f}, {s_cf.max().item():.3f}]")
    assert s_cf.shape == (B, 1)

    # Score should differ after masking (non-trivial model response)
    diff = (s - s_cf).abs().mean().item()
    print(f"                 mean |score - score_masked| = {diff:.4f}")

    # Gradient flow through both paths
    (s.mean() + s_cf.mean()).backward()
    assert user_emb.grad is not None and item_emb.grad is not None
    print("\nOK: forward, forward_masked, and backward all pass.")
