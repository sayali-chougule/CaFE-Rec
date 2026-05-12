# PyTorch Dataset / DataLoader wrapping interactions, aspects, and reference text.
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_user_pos_map(df: pd.DataFrame) -> Dict[int, set]:
    """user_idx -> set of item_idx the user has interacted with."""
    return (
        df.groupby("user_idx")["item_idx"]
        .apply(lambda s: set(int(x) for x in s))
        .to_dict()
    )


def _sample_negatives(
    rng: np.random.RandomState,
    num_items: int,
    exclude: set,
    n: int,
) -> List[int]:
    """Rejection-sample `n` distinct item ids not in `exclude`."""
    out: List[int] = []
    seen = set(exclude)
    while len(out) < n:
        # Over-sample to reduce python-loop overhead.
        cand = rng.randint(0, num_items, size=max(n * 2, 32))
        for c in cand:
            c = int(c)
            if c in seen:
                continue
            out.append(c)
            seen.add(c)
            if len(out) == n:
                break
    return out


# ---------------------------------------------------------------------------
# 1. InteractionDataset (BPR training)
# ---------------------------------------------------------------------------
class InteractionDataset(Dataset):
    """(user, pos_item, neg_item) triples for BPR training."""

    def __init__(self, train_df: pd.DataFrame, num_items: int, seed: int = 0):
        self.num_items = int(num_items)
        self.users = train_df["user_idx"].to_numpy(dtype=np.int64)
        self.pos_items = train_df["item_idx"].to_numpy(dtype=np.int64)
        self.user_pos = _build_user_pos_map(train_df)
        # Each worker gets its own RNG; seeded from worker id in worker_init_fn.
        self._rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        user = int(self.users[idx])
        pos = int(self.pos_items[idx])
        pos_set = self.user_pos.get(user, set())
        # Fast path: one draw usually suffices.
        while True:
            neg = int(self._rng.randint(0, self.num_items))
            if neg not in pos_set:
                break
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


def interaction_collate_fn(batch):
    users, pos, neg = zip(*batch)
    return (
        torch.stack(users, dim=0),
        torch.stack(pos, dim=0),
        torch.stack(neg, dim=0),
    )


# ---------------------------------------------------------------------------
# 2. ExplanationDataset (text generation training)
# ---------------------------------------------------------------------------
class ExplanationDataset(Dataset):
    """(user, item, aspect_labels, tokenized explanation) per row."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        aspect2idx: Dict[str, int],
        max_length: int = 128,
        text_column: str = "review_text",
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.aspect2idx = aspect2idx
        self.num_aspects = len(aspect2idx)
        self.max_length = int(max_length)
        self.text_column = text_column
        self._has_labels = "aspect_labels" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        user = int(row["user_idx"])
        item = int(row["item_idx"])
        text = str(row[self.text_column]) if row[self.text_column] else ""

        if self._has_labels:
            labels = np.asarray(row["aspect_labels"], dtype=np.float32)
        else:
            labels = np.zeros(self.num_aspects, dtype=np.float32)

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "user_idx": torch.tensor(user, dtype=torch.long),
            "item_idx": torch.tensor(item, dtype=torch.long),
            "aspect_labels": torch.from_numpy(labels),
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def explanation_collate_fn(batch: List[Dict[str, torch.Tensor]]):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


# ---------------------------------------------------------------------------
# 3. EvaluationDataset (ranking metrics: Recall@K, NDCG@K)
# ---------------------------------------------------------------------------
class EvaluationDataset(Dataset):
    """(user, pos_item, [neg_items x num_negatives]) for held-out ranking.

    Negatives are precomputed with a per-user seed so every evaluation run
    scores the same candidate slate — essential for reproducible metrics.
    Training positives (and the current test positive) are excluded from
    the candidate pool.
    """

    def __init__(
        self,
        test_df: pd.DataFrame,
        train_df: pd.DataFrame,
        num_items: int,
        num_negatives: int = 100,
        seed: int = 42,
        val_df: Optional[pd.DataFrame] = None,
    ):
        self.test_df = test_df.reset_index(drop=True)
        self.num_items = int(num_items)
        self.num_negatives = int(num_negatives)

        # Exclude everything seen in train (and optionally val) from negatives.
        exclude_map = _build_user_pos_map(train_df)
        if val_df is not None:
            val_map = _build_user_pos_map(val_df)
            for u, s in val_map.items():
                exclude_map.setdefault(u, set()).update(s)

        users = self.test_df["user_idx"].to_numpy(dtype=np.int64)
        pos = self.test_df["item_idx"].to_numpy(dtype=np.int64)

        self.users = users
        self.pos_items = pos
        self.neg_items = np.empty(
            (len(self.test_df), self.num_negatives), dtype=np.int64
        )

        for i in range(len(self.test_df)):
            u = int(users[i])
            p = int(pos[i])
            exclude = exclude_map.get(u, set()) | {p}
            rng = np.random.RandomState(seed + u)
            self.neg_items[i] = _sample_negatives(
                rng, self.num_items, exclude, self.num_negatives
            )

    def __len__(self) -> int:
        return len(self.test_df)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(int(self.users[idx]), dtype=torch.long),
            torch.tensor(int(self.pos_items[idx]), dtype=torch.long),
            torch.from_numpy(self.neg_items[idx]).to(torch.long),
        )


def evaluation_collate_fn(batch):
    users, pos, negs = zip(*batch)
    return (
        torch.stack(users, dim=0),
        torch.stack(pos, dim=0),
        torch.stack(negs, dim=0),
    )
