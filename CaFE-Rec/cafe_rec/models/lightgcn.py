# LightGCN backbone: learns user and item embeddings from the interaction graph.
from __future__ import annotations

from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    """Light Graph Convolutional Network (He et al., SIGIR 2020).

    Learns user and item embeddings via repeated neighborhood aggregation on
    the user-item bipartite graph. Unlike vanilla GCN, LightGCN omits
    per-layer weight matrices and non-linear activations ("light"), leaving
    only symmetric-normalized propagation:

        e_u^(k+1) = sum_{i in N(u)} 1/sqrt(|N(u)| |N(i)|) * e_i^(k)

    The final representation is the mean of the layer-0..layer-K embeddings,
    which acts as a self-implemented form of residual/skip connection.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_users = int(num_users)
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------
    def forward(
        self, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run K rounds of LightGCN propagation.

        Args:
            edge_index: LongTensor [2, E] of an *undirected* bipartite graph
                where user nodes are 0..num_users-1 and item nodes are
                num_users..num_users+num_items-1. Both directions
                (user->item and item->user) must be present.

        Returns:
            (user_emb, item_emb), each of shape [num_users/items, dim],
            being the mean of the 0..num_layers propagated embeddings.
        """
        num_nodes = self.num_users + self.num_items
        device = edge_index.device
        row, col = edge_index[0], edge_index[1]

        # Symmetric normalization weights: 1 / sqrt(deg_u * deg_v)
        deg = torch.zeros(num_nodes, device=device, dtype=torch.float)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]  # [E]

        # Stack user and item embeddings into one node table.
        e0 = torch.cat(
            [self.user_embedding.weight, self.item_embedding.weight], dim=0
        )
        all_embs = [e0]
        e = e0
        for _ in range(self.num_layers):
            e = self._propagate(e, row, col, edge_weight, num_nodes)
            if self.dropout > 0 and self.training:
                e = F.dropout(e, p=self.dropout)
            all_embs.append(e)

        e_final = torch.stack(all_embs, dim=0).mean(dim=0)
        user_emb = e_final[: self.num_users]
        item_emb = e_final[self.num_users:]
        return user_emb, item_emb

    @staticmethod
    def _propagate(
        e: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """One layer of weighted neighbor aggregation (no weight matrix)."""
        msg = edge_weight.unsqueeze(-1) * e[col]  # [E, dim]
        out = torch.zeros(num_nodes, e.size(-1), device=e.device, dtype=e.dtype)
        out.scatter_add_(
            0, row.unsqueeze(-1).expand(-1, e.size(-1)), msg
        )
        return out

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @staticmethod
    def get_recommendation_scores(
        user_emb: torch.Tensor, item_emb: torch.Tensor
    ) -> torch.Tensor:
        """Dot-product score for paired user/item embeddings.

        Args:
            user_emb: [B, D]
            item_emb: [B, D]
        Returns:
            [B] tensor of preference scores.
        """
        return (user_emb * item_emb).sum(dim=-1)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    @staticmethod
    def build_edge_index(
        train_df: pd.DataFrame,
        num_users: int,
        num_items: int,
    ) -> torch.LongTensor:
        """Build an undirected bipartite edge_index from a training DataFrame.

        Item node ids are offset by `num_users`, so item 0 becomes node
        `num_users` in the combined node space. Both user->item and
        item->user edges are emitted, yielding `2 * num_interactions` edges.
        """
        u = torch.tensor(
            train_df["user_idx"].to_numpy(), dtype=torch.long
        )
        i = torch.tensor(
            train_df["item_idx"].to_numpy(), dtype=torch.long
        ) + int(num_users)
        row = torch.cat([u, i], dim=0)
        col = torch.cat([i, u], dim=0)
        edge_index = torch.stack([row, col], dim=0)
        return edge_index


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    num_users, num_items = 20, 30
    # Fake interactions
    df = pd.DataFrame({
        "user_idx": torch.randint(0, num_users, (200,)).tolist(),
        "item_idx": torch.randint(0, num_items, (200,)).tolist(),
    })

    edge_index = LightGCN.build_edge_index(df, num_users, num_items)
    print(f"edge_index shape: {tuple(edge_index.shape)} "
          f"(expected [2, {2 * len(df)}])")

    model = LightGCN(num_users, num_items, embedding_dim=16, num_layers=3)
    model.eval()

    user_emb, item_emb = model(edge_index)
    print(f"user_emb: {tuple(user_emb.shape)}  item_emb: {tuple(item_emb.shape)}")
    assert user_emb.shape == (num_users, 16)
    assert item_emb.shape == (num_items, 16)

    # Score a small batch
    batch_users = torch.tensor([0, 1, 2, 3])
    batch_items = torch.tensor([5, 6, 7, 8])
    scores = LightGCN.get_recommendation_scores(
        user_emb[batch_users], item_emb[batch_items]
    )
    print(f"scores shape: {tuple(scores.shape)}  values: {scores.tolist()}")
    assert scores.shape == (4,)

    # Gradient check
    model.train()
    user_emb, item_emb = model(edge_index)
    loss = LightGCN.get_recommendation_scores(
        user_emb[batch_users], item_emb[batch_items]
    ).mean()
    loss.backward()
    assert model.user_embedding.weight.grad is not None
    print("OK: forward, scoring, and backward all pass.")
