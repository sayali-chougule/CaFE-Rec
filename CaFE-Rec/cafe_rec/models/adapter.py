# Linear adapter mapping selected-aspect representation into Flan-T5 prefix embeddings.
from __future__ import annotations

import torch
import torch.nn as nn


class AspectToT5Adapter(nn.Module):
    """Project (selected_aspects, user_emb, item_emb) into a small bank of
    soft-prefix embeddings that Flan-T5's encoder can consume.

    The adapter is the only learned bridge between CaFE-Rec's recommender
    backbone and the frozen/finetuned Flan-T5. At generation time these
    prefix tokens are prepended to the encoder's input embeddings, giving
    Flan-T5 access to *exactly* the information that the ConstrainedScorer
    also sees -- which is what keeps the generated explanation faithful
    to the recommendation.
    """

    def __init__(
        self,
        aspect_dim: int = 50,
        user_item_dim: int = 128,
        t5_hidden_dim: int = 768,
        num_prefix_tokens: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.aspect_dim = int(aspect_dim)
        self.user_item_dim = int(user_item_dim)
        self.t5_hidden_dim = int(t5_hidden_dim)
        self.num_prefix_tokens = int(num_prefix_tokens)

        self.project_aspects = nn.Linear(self.aspect_dim, self.t5_hidden_dim)
        self.project_user_item = nn.Linear(
            self.user_item_dim, self.t5_hidden_dim
        )
        self.combine = nn.Linear(
            2 * self.t5_hidden_dim,
            self.num_prefix_tokens * self.t5_hidden_dim,
        )
        self.dropout = nn.Dropout(p=float(dropout))

        for m in (self.project_aspects, self.project_user_item, self.combine):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        selected_aspects: torch.Tensor,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Build a [B, num_prefix_tokens, t5_hidden_dim] prefix bank.

        Args:
            selected_aspects: [B, aspect_dim] soft/hard weights from AspectSelector.
            user_emb: [B, D_u]
            item_emb: [B, D_i]   (D_u + D_i must equal user_item_dim)
        Returns:
            prefix_embeddings: [B, num_prefix_tokens, t5_hidden_dim].
        """
        B = selected_aspects.size(0)
        ui = torch.cat([user_emb, item_emb], dim=-1)          # [B, D_ui]
        a_proj = self.project_aspects(selected_aspects)       # [B, H]
        ui_proj = self.project_user_item(ui)                  # [B, H]
        combined = torch.cat([a_proj, ui_proj], dim=-1)       # [B, 2H]
        combined = self.dropout(combined)
        flat = self.combine(combined)                         # [B, P*H]
        prefix = flat.view(B, self.num_prefix_tokens, self.t5_hidden_dim)
        return prefix


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, D_u, D_i, N, H, P = 4, 64, 64, 50, 768, 10
    aspects = torch.rand(B, N)
    user_emb = torch.randn(B, D_u, requires_grad=True)
    item_emb = torch.randn(B, D_i, requires_grad=True)

    adapter = AspectToT5Adapter(
        aspect_dim=N,
        user_item_dim=D_u + D_i,
        t5_hidden_dim=H,
        num_prefix_tokens=P,
    )
    print(adapter)

    prefix = adapter(aspects, user_emb, item_emb)
    print(f"\nprefix shape: {tuple(prefix.shape)}  "
          f"(expected ({B}, {P}, {H}))")
    assert prefix.shape == (B, P, H)

    # Gradient flow: simulate downstream T5 using prefix.
    loss = prefix.pow(2).mean()
    loss.backward()
    assert user_emb.grad is not None and item_emb.grad is not None
    assert adapter.project_aspects.weight.grad is not None
    assert adapter.combine.weight.grad is not None

    # Parameter count sanity
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"total params: {n_params:,}")
    print("OK: forward, reshape, and backward all pass.")
