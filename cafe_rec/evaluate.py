# Evaluation entrypoint: rec / text / fidelity metrics, plus a baseline comparison table.
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, CONFIG
from data.dataset import (
    EvaluationDataset,
    ExplanationDataset,
    evaluation_collate_fn,
    explanation_collate_fn,
)
from evaluation.fidelity_metrics import (
    counterfactual_fidelity_percentage,
    run_full_fidelity_evaluation,
)
from evaluation.rec_metrics import (
    evaluate_recommendations,
    ndcg_at_k,
    recall_at_k,
)
from evaluation.text_metrics import (
    compute_bertscore,
    compute_bleu,
    compute_rouge,
)
from models.cafe_rec import CaFeRec


# ---------------------------------------------------------------------------
# 1. Load checkpoint
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Config,
    train_df: pd.DataFrame,
    vocab: Dict[str, Any],
) -> CaFeRec:
    """Reconstruct CaFeRec from a training checkpoint and set eval mode."""
    num_users = int(train_df["user_idx"].max()) + 1
    num_items = int(train_df["item_idx"].max()) + 1
    model = CaFeRec(config, num_users, num_items, vocab["aspects"])
    model.to(torch.device(config.device))
    model.build_graph(train_df)

    ckpt = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[ckpt] loaded {checkpoint_path} "
          f"(epoch {ckpt.get('epoch', '?')}, "
          f"val={ckpt.get('val_score', float('nan')):.4f})")
    return model


# ---------------------------------------------------------------------------
# Per-example metrics — needed for Wilcoxon significance testing
# ---------------------------------------------------------------------------
@torch.no_grad()
def per_example_rec_metrics(
    model: CaFeRec, dataset: EvaluationDataset, config: Config,
    batch_size: int = 64, k_list: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """Recall@K / NDCG@K per row (needed for paired significance tests)."""
    model.eval()
    device = next(model.parameters()).device
    ks = k_list or list(config.top_k_list)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=evaluation_collate_fn,
    )
    all_u, all_i = model.get_embeddings()

    per = {f"Recall@{k}": [] for k in ks}
    per.update({f"NDCG@{k}": [] for k in ks})

    for users, pos, negs in tqdm(loader, desc="per-example rec"):
        users = users.to(device); pos = pos.to(device); negs = negs.to(device)
        B, Nneg = negs.shape
        cands = torch.cat([pos.unsqueeze(1), negs], dim=1)
        ncand = cands.size(1)

        u_exp = (
            all_u[users].unsqueeze(1).expand(-1, ncand, -1)
            .reshape(B * ncand, -1)
        )
        i_exp = all_i[cands.reshape(-1)]
        sel, _ = model.aspect_selector(u_exp, i_exp, hard=True)
        scores = model.constrained_scorer(u_exp, i_exp, sel).view(B, ncand)
        scores_np = scores.cpu().numpy()
        for i in range(B):
            for k in ks:
                per[f"Recall@{k}"].append(recall_at_k(scores_np[i], 0, k))
                per[f"NDCG@{k}"].append(ndcg_at_k(scores_np[i], 0, k))

    return {k: np.asarray(v) for k, v in per.items()}


@torch.no_grad()
def per_example_cf_flags(
    model: CaFeRec, dataset: EvaluationDataset, config: Config,
    k: int = 10, batch_size: int = 64,
) -> np.ndarray:
    """Binary per-row flag: 1 if the explanation flipped the top-K for this
    example (pos was in top-K factually, not counterfactually)."""
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=evaluation_collate_fn,
    )
    all_u, all_i = model.get_embeddings()
    flags: List[int] = []

    for users, pos, negs in tqdm(loader, desc="per-example CF"):
        users = users.to(device); pos = pos.to(device); negs = negs.to(device)
        B, Nneg = negs.shape
        cands = torch.cat([pos.unsqueeze(1), negs], dim=1)
        ncand = cands.size(1)

        u_emb = all_u[users]
        sel_pos, _ = model.aspect_selector(u_emb, all_i[pos], hard=True)
        topk = sel_pos.topk(
            min(config.num_selected_aspects, sel_pos.size(-1)), dim=-1
        ).indices
        expl = torch.zeros_like(sel_pos); expl.scatter_(1, topk, 1.0)

        u_exp = (
            u_emb.unsqueeze(1).expand(-1, ncand, -1).reshape(B * ncand, -1)
        )
        i_exp = all_i[cands.reshape(-1)]
        sel_all, _ = model.aspect_selector(u_exp, i_exp, hard=True)
        fact = model.constrained_scorer(u_exp, i_exp, sel_all).view(B, ncand)
        mask_exp = (
            expl.unsqueeze(1).expand(-1, ncand, -1).reshape(B * ncand, -1)
        )
        cf = model.constrained_scorer(
            u_exp, i_exp, sel_all * (1.0 - mask_exp)
        ).view(B, ncand)

        fact_has = (fact.topk(k, dim=-1).indices == 0).any(dim=-1)
        cf_has = (cf.topk(k, dim=-1).indices == 0).any(dim=-1)
        flipped = (fact_has & ~cf_has).cpu().numpy().astype(np.int32)
        # "eligible" = pos was in top-K factually; for rows where it was not,
        # CF% is undefined — we mark as -1 so the caller can filter.
        elig = fact_has.cpu().numpy()
        for f, e in zip(flipped, elig):
            flags.append(int(f) if bool(e) else -1)
    return np.asarray(flags, dtype=np.int32)


# ---------------------------------------------------------------------------
# 2. Evaluate everything for our model
# ---------------------------------------------------------------------------
@torch.no_grad()
def _generate_texts(
    model: CaFeRec,
    df: pd.DataFrame,
    max_length: int = 64,
    sample_limit: Optional[int] = 500,
) -> Tuple[List[str], List[str]]:
    """Decode Flan-T5 explanations for up to `sample_limit` test rows."""
    model.eval()
    device = next(model.parameters()).device
    if sample_limit is not None and len(df) > sample_limit:
        df = df.sample(
            n=sample_limit, random_state=0, replace=False
        ).reset_index(drop=True)

    preds, refs = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="generating text"):
        u = int(row["user_idx"]); i = int(row["item_idx"])
        try:
            text, _ = model.generate_explanation(u, i, max_length=max_length)
        except Exception:
            text = ""
        preds.append(text)
        refs.append(str(row.get("review_text", "")))
    return preds, refs


def evaluate_model(
    model: CaFeRec,
    test_dataset: EvaluationDataset,
    test_df: pd.DataFrame,
    config: Config,
    text_sample_limit: Optional[int] = 500,
    cf_k: int = 10,
) -> Dict[str, Any]:
    """Full evaluation: rec + fidelity + generated-text metrics."""
    rec = evaluate_recommendations(
        model, test_dataset, config, batch_size=max(config.batch_size, 32)
    )
    fid = run_full_fidelity_evaluation(
        model, test_dataset, config,
        batch_size=max(config.batch_size, 32), cf_k=cf_k,
    )

    preds, refs = _generate_texts(
        model, test_df, max_length=64, sample_limit=text_sample_limit
    )
    text_metrics: Dict[str, float] = {}
    try:
        text_metrics.update(compute_bleu(preds, refs))
    except Exception as e:
        print(f"[text] BLEU skipped: {e}")
    try:
        text_metrics.update(compute_rouge(preds, refs))
    except Exception as e:
        print(f"[text] ROUGE skipped: {e}")
    try:
        text_metrics.update(
            compute_bertscore(preds, refs, device=str(config.device))
        )
    except Exception as e:
        print(f"[text] BERTScore skipped: {e}")

    return {
        **{k: float(v) for k, v in rec.items()},
        **{k: float(v) for k, v in fid.items()},
        **{k: float(v) for k, v in text_metrics.items()},
        "n_text_samples": len(preds),
    }


# ---------------------------------------------------------------------------
# 3. Formatted comparison table
# ---------------------------------------------------------------------------
_TABLE_COLUMNS = [
    ("Recall@10", "Recall@10"),
    ("NDCG@10", "NDCG@10"),
    ("BLEU-4", "BLEU-4"),
    ("Comp.", "comprehensiveness"),
    ("Suff.", "sufficiency"),
    ("CF%", "CF@10"),
]


def _fmt_cell(value: Any) -> str:
    if value is None:
        return "  N/A  "
    if isinstance(value, str):
        return f"{value:>7}"
    try:
        return f"{float(value):7.4f}"
    except (TypeError, ValueError):
        return f"{str(value):>7}"


def format_results_table(
    all_results: Dict[str, Dict[str, Any]],
    ours_key: str = "CaFE-Rec(ours)",
    significance: Optional[Dict[str, bool]] = None,
) -> str:
    """Pretty-print a model-vs-metric comparison table.

    `all_results` is {model_name: {metric: value}}; `significance` maps
    metric -> bool indicating whether OUR result is statistically
    significant over the strongest baseline (adds a trailing * to the cell).
    """
    header = f"{'Model':<16}| " + " | ".join(
        f"{name:>7}" for name, _ in _TABLE_COLUMNS
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for model_name, res in all_results.items():
        row = [f"{model_name:<16}"]
        for display, key in _TABLE_COLUMNS:
            val = res.get(key)
            cell = _fmt_cell(val)
            if (model_name == ours_key and significance
                    and significance.get(key, False)):
                cell = cell.rstrip() + "*"
                cell = f"{cell:>7}"
            row.append(cell)
        lines.append(row[0] + "| " + " | ".join(row[1:]))

    lines.append(sep)
    if significance and any(significance.values()):
        lines.append("* statistically significant (p < 0.05, Wilcoxon test)")
    out = "\n".join(lines)
    print(out)
    return out


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------
def _wilcoxon(ours: np.ndarray, theirs: np.ndarray) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank. Returns (statistic, p_value)."""
    from scipy.stats import wilcoxon  # type: ignore
    n = min(len(ours), len(theirs))
    a, b = ours[:n], theirs[:n]
    diff = a - b
    if np.all(diff == 0):
        return 0.0, 1.0
    try:
        stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="greater")
        return float(stat), float(p)
    except ValueError as e:
        print(f"[wilcoxon] skipped ({e})")
        return float("nan"), float("nan")


def run_significance_tests(
    ours_per: Dict[str, np.ndarray],
    baseline_per: Dict[str, Dict[str, np.ndarray]],
    metrics: Tuple[str, ...] = ("Recall@10", "CF@10"),
    strongest: str = "XRec",
) -> Dict[str, bool]:
    """Compare CaFE-Rec against the strongest baseline, per-metric."""
    results: Dict[str, bool] = {}
    if strongest not in baseline_per:
        return {m: False for m in metrics}
    for m in metrics:
        a = ours_per.get(m)
        b = baseline_per[strongest].get(m)
        if a is None or b is None:
            results[m] = False
            continue
        if m == "CF@10":
            a = a[a >= 0]; b = b[b >= 0]
        _, p = _wilcoxon(np.asarray(a, dtype=float),
                         np.asarray(b, dtype=float))
        results[m] = bool(p < 0.05)
        print(f"[wilcoxon] {m}: ours vs {strongest}  p={p:.4g}  "
              f"{'SIGNIFICANT' if results[m] else 'n.s.'}")
    return results


# ---------------------------------------------------------------------------
# 4. Qualitative example explanations
# ---------------------------------------------------------------------------
def generate_example_explanations(
    model: CaFeRec,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    config: Config,
    n_examples: int = 5,
) -> List[Dict[str, Any]]:
    """Print and return sample explanations plus one or two failure cases."""
    from evaluation.fidelity_metrics import sufficiency_score

    model.eval()
    device = next(model.parameters()).device
    rng = np.random.RandomState(7)
    user_history = (
        train_df.groupby("user_idx")["item_idx"]
        .apply(list).to_dict()
    )

    picks = rng.choice(len(test_df), size=min(n_examples + 2, len(test_df)),
                       replace=False)
    examples: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("Example explanations")
    print("=" * 70)
    for rank, idx in enumerate(picks):
        row = test_df.iloc[int(idx)]
        u, i = int(row["user_idx"]), int(row["item_idx"])
        text, top_aspects = model.generate_explanation(u, i, max_length=64)
        suff = sufficiency_score(
            model,
            torch.tensor([u], dtype=torch.long, device=device),
            torch.tensor([i], dtype=torch.long, device=device),
            None, config,
        )
        hist = user_history.get(u, [])[:5]
        kind = "example" if rank < n_examples else "failure-candidate"
        ex = {
            "kind": kind,
            "user_idx": u, "item_idx": i,
            "history_first5": hist,
            "generated_explanation": text,
            "top_aspects": top_aspects,
            "sufficiency": float(suff),
            "reference": str(row.get("review_text", ""))[:180],
        }
        examples.append(ex)
        print(f"\n[{kind}] user={u} item={i}")
        print(f"  history (first 5 items): {hist}")
        print(f"  top aspects: {top_aspects}")
        print(f"  explanation: {text!r}")
        print(f"  sufficiency: {suff:.3f}")
        print(f"  reference : {ex['reference']!r}")
    return examples


# ---------------------------------------------------------------------------
# Baseline loading
# ---------------------------------------------------------------------------
def load_baseline_results(
    path: Optional[str], dataset_name: str
) -> Dict[str, Dict[str, Any]]:
    """Accept either a single JSON file or a directory of <model>.json files."""
    if path is None:
        return {}
    results: Dict[str, Dict[str, Any]] = {}
    if os.path.isdir(path):
        for f in sorted(glob.glob(os.path.join(path, "*.json"))):
            name = os.path.splitext(os.path.basename(f))[0]
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            # Support either {dataset: {...}} or {...} directly.
            if dataset_name in data:
                results[name] = data[dataset_name]
            else:
                results[name] = data
    elif os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for name, payload in data.items():
            if isinstance(payload, dict) and dataset_name in payload:
                results[name] = payload[dataset_name]
            else:
                results[name] = payload
    else:
        print(f"[baselines] path not found: {path}")
    return results


# ---------------------------------------------------------------------------
# 5. CLI main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CaFE-Rec.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to a .pt checkpoint from train.py.")
    p.add_argument("--dataset", default="amazon_movies")
    p.add_argument("--processed_dir", default=None)
    p.add_argument("--baselines", default="./results/baselines",
                   help="JSON file or directory of baseline results.")
    p.add_argument("--out_dir", default="./results")
    p.add_argument("--text_limit", type=int, default=500,
                   help="Max test rows to generate explanations for.")
    p.add_argument("--cf_k", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = CONFIG
    if args.processed_dir:
        config = replace(config, processed_dir=args.processed_dir)
    config = replace(config, dataset_name=args.dataset)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    d = config.processed_dir
    with open(os.path.join(d, "train.pkl"), "rb") as f:
        train_df = pickle.load(f)
    with open(os.path.join(d, "test.pkl"), "rb") as f:
        test_df = pickle.load(f)
    with open(os.path.join(d, "aspect_vocab.json"), "r",
              encoding="utf-8") as f:
        vocab = json.load(f)
    num_items = int(train_df["item_idx"].max()) + 1

    # Build eval dataset from test (exclude train positives from negatives)
    test_ds = EvaluationDataset(
        test_df, train_df, num_items=num_items,
        num_negatives=config.num_neg_samples, seed=42,
    )

    # Model
    model = load_model_from_checkpoint(
        args.checkpoint, config, train_df, vocab
    )

    # Our results
    print("\n[eval] running full evaluation on test set...")
    ours = evaluate_model(
        model, test_ds, test_df, config,
        text_sample_limit=args.text_limit, cf_k=args.cf_k,
    )

    # Per-example scores for significance testing
    ours_per_rec = per_example_rec_metrics(
        model, test_ds, config, k_list=[10]
    )
    ours_cf = per_example_cf_flags(model, test_ds, config, k=args.cf_k)
    ours_per = {
        "Recall@10": ours_per_rec["Recall@10"],
        "CF@10": ours_cf,
    }
    np.savez(
        os.path.join(args.out_dir, "cafe_rec_per_example.npz"),
        **{k: v for k, v in ours_per.items()},
    )

    # Baselines (aggregate + optional per-example)
    baselines = load_baseline_results(args.baselines, config.dataset_name)

    baseline_per: Dict[str, Dict[str, np.ndarray]] = {}
    if os.path.isdir(args.baselines):
        for name in baselines:
            npz = os.path.join(args.baselines, f"{name}_per_example.npz")
            if os.path.exists(npz):
                z = np.load(npz)
                baseline_per[name] = {k: z[k] for k in z.files}

    significance = run_significance_tests(
        ours_per, baseline_per, metrics=("Recall@10", "CF@10"),
        strongest="XRec",
    ) if baseline_per else {m: False for m in ("Recall@10", "CF@10")}

    # Compose the table
    all_results = {**baselines, "CaFE-Rec(ours)": ours}
    table_str = format_results_table(
        all_results, ours_key="CaFE-Rec(ours)", significance=significance,
    )

    # Qualitative examples
    examples = generate_example_explanations(
        model, test_df, train_df, config, n_examples=5
    )

    # Save everything
    out = {
        "dataset": config.dataset_name,
        "checkpoint": args.checkpoint,
        "config": config.as_dict(),
        "results": all_results,
        "significance": {k: bool(v) for k, v in significance.items()},
        "examples": examples,
        "table": table_str,
    }
    out_path = os.path.join(args.out_dir, "results_table.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
