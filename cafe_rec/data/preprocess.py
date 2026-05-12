# Raw review files -> cleaned (user, item, rating, text) interactions; train/val/test splits.
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm


# Amazon Reviews 2023 category aliases -> HuggingFace config names.
AMAZON_CATEGORY_ALIASES = {
    "movies_tv": "Movies_and_TV",
    "movies_and_tv": "Movies_and_TV",
    "books": "Books",
    "electronics": "Electronics",
    "cds_vinyl": "CDs_and_Vinyl",
    "video_games": "Video_Games",
}

REQUIRED_COLS = ["user_id", "item_id", "rating", "review_text", "timestamp"]


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_amazon_reviews(
    data_dir: str,
    category: str = "movies_tv",
) -> pd.DataFrame:
    """Load Amazon Reviews 2023 for a given category.

    Tries local JSONL(.gz) files in data_dir first, then falls back to the
    McAuley-Lab/Amazon-Reviews-2023 HuggingFace dataset. Returns a DataFrame
    with columns: user_id, item_id, rating, review_text, timestamp.
    """
    hf_cat = AMAZON_CATEGORY_ALIASES.get(category.lower(), category)
    local = _find_local_jsonl(data_dir, hf_cat)

    if local is not None:
        print(f"[load] reading local file: {local}")
        rows = list(_iter_jsonl(local))
    else:
        print(f"[load] local file not found; fetching HuggingFace "
              f"McAuley-Lab/Amazon-Reviews-2023 / raw_review_{hf_cat}")
        rows = _load_from_huggingface(hf_cat)

    df = pd.DataFrame(rows)
    df = _normalize_columns(df)
    df = df[REQUIRED_COLS].dropna(subset=["user_id", "item_id", "rating"])
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["review_text"] = df["review_text"].fillna("").astype(str)
    print(f"[load] loaded {len(df):,} raw interactions "
          f"({df['user_id'].nunique():,} users, "
          f"{df['item_id'].nunique():,} items)")
    return df.reset_index(drop=True)


def _find_local_jsonl(data_dir: str, hf_cat: str):
    p = Path(data_dir)
    if not p.exists():
        return None
    for name in (f"{hf_cat}.jsonl.gz", f"{hf_cat}.jsonl",
                 f"{hf_cat}.json.gz", f"{hf_cat}.json"):
        f = p / name
        if f.exists():
            return f
    return None


def _iter_jsonl(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in tqdm(fh, desc=f"reading {path.name}"):
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_from_huggingface(hf_cat: str):
    from datasets import load_dataset
    ds = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        f"raw_review_{hf_cat}",
        split="full",
        trust_remote_code=True,
    )
    return list(tqdm(ds, desc="streaming HF rows"))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Amazon Reviews 2023 field names to our canonical schema."""
    rename = {
        "asin": "item_id",
        "parent_asin": "item_id",
        "reviewerID": "user_id",
        "reviewText": "review_text",
        "text": "review_text",
        "overall": "rating",
        "unixReviewTime": "timestamp",
    }
    for src, dst in rename.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = None
    return df


# ---------------------------------------------------------------------------
# 2. k-core filtering
# ---------------------------------------------------------------------------
def apply_k_core_filter(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Iteratively drop users/items with < k interactions until stable."""
    prev = -1
    it = 0
    while len(df) != prev:
        prev = len(df)
        it += 1
        u_counts = df["user_id"].value_counts()
        i_counts = df["item_id"].value_counts()
        keep_u = u_counts[u_counts >= k].index
        keep_i = i_counts[i_counts >= k].index
        df = df[df["user_id"].isin(keep_u) & df["item_id"].isin(keep_i)]
    print(f"After k-core filtering: {df['user_id'].nunique():,} users, "
          f"{df['item_id'].nunique():,} items, {len(df):,} interactions "
          f"(converged in {it} iterations)")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. ID mappings
# ---------------------------------------------------------------------------
def create_id_mappings(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict, Dict, Dict, Dict]:
    """Remap user/item ids to contiguous [0, N) integers."""
    users = sorted(df["user_id"].unique())
    items = sorted(df["item_id"].unique())
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {it: i for i, it in enumerate(items)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: it for it, i in item2idx.items()}

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user2idx).astype("int64")
    df["item_idx"] = df["item_id"].map(item2idx).astype("int64")
    print(f"[mapping] {len(user2idx):,} users -> [0,{len(user2idx)}); "
          f"{len(item2idx):,} items -> [0,{len(item2idx)})")
    return df, user2idx, item2idx, idx2user, idx2item


# ---------------------------------------------------------------------------
# 4. Split
# ---------------------------------------------------------------------------
def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split: sort by timestamp then slice."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
    print(f"[split] train={len(train_df):,}  val={len(val_df):,}  "
          f"test={len(test_df):,}")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mappings: Dict[str, Dict],
    save_dir: str,
) -> None:
    """Pickle the three splits; dump id mappings as JSON."""
    os.makedirs(save_dir, exist_ok=True)
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        path = os.path.join(save_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[save] {path}  ({len(df):,} rows)")

    # JSON keys must be strings
    mapping_json = {
        k: {str(kk): vv for kk, vv in v.items()} for k, v in mappings.items()
    }
    map_path = os.path.join(save_dir, "mappings.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping_json, f)
    print(f"[save] {map_path}")


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------
def main(
    data_dir: str = "./data/raw",
    processed_dir: str = "./data/processed",
    category: str = "movies_tv",
    k: int = 5,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> None:
    print("=" * 60)
    print(f"CaFE-Rec preprocessing: Amazon Reviews 2023 ({category})")
    print("=" * 60)

    df = load_amazon_reviews(data_dir, category=category)
    df = apply_k_core_filter(df, k=k)
    df, user2idx, item2idx, idx2user, idx2item = create_id_mappings(df)
    train_df, val_df, test_df = train_val_test_split(
        df, train_ratio, val_ratio, test_ratio
    )
    save_processed_data(
        train_df, val_df, test_df,
        mappings={
            "user2idx": user2idx,
            "item2idx": item2idx,
            "idx2user": idx2user,
            "idx2item": idx2item,
        },
        save_dir=processed_dir,
    )
    print("=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/raw")
    parser.add_argument("--processed_dir", default="./data/processed")
    parser.add_argument("--category", default="movies_tv")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        category=args.category,
        k=args.k,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
