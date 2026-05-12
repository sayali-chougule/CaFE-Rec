"""End-to-end smoke test for the CaFE-Rec pipeline on synthetic data.

Run this BEFORE any real training. It exercises every module in order
using 100 fake users / 200 items / 1000 interactions, prints PASS/FAIL
per section, and aborts early if a stage fails.

Target runtime: under 60s on Apple Silicon (dominated by the first-time
Flan-T5-Base download in stage 6 — cached on later runs).
"""
from __future__ import annotations

import random
import sys
import time
import traceback
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class Stage:
    name: str
    passed: bool
    duration: float
    detail: str = ""


_RESULTS: List[Stage] = []


def _run(name: str, fn):
    t0 = time.time()
    print(f"\n── {name} " + "─" * (60 - len(name)))
    try:
        detail = fn() or ""
        dt = time.time() - t0
        _RESULTS.append(Stage(name, True, dt, str(detail)))
        print(f"✓ PASS  ({dt:.2f}s)")
    except Exception as e:
        dt = time.time() - t0
        tb = traceback.format_exc()
        _RESULTS.append(Stage(name, False, dt, f"{e}\n{tb}"))
        print(f"✗ FAIL  ({dt:.2f}s)\n{tb}")
        return False
    return True


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
NUM_USERS = 100
NUM_ITEMS = 200
NUM_INTERACTIONS = 1000
ASPECTS = [
    "action", "romance", "comedy", "sci-fi", "thriller",
    "drama", "animation", "horror", "documentary", "fantasy",
]
REVIEW_TEMPLATES = [
    "Great {a} movie",
    "Loved the {a} scenes",
    "Solid {a} throughout",
    "Weak {a} but watchable",
    "Brilliant {a} direction",
    "Terrible {a}, skip it",
    "Nice {a} and great acting",
    "The {a} was a highlight",
]


def _build_fake_data(seed: int = 0):
    rng = random.Random(seed)
    rows = []
    for _ in range(NUM_INTERACTIONS):
        u = rng.randrange(NUM_USERS)
        i = rng.randrange(NUM_ITEMS)
        a = rng.choice(ASPECTS)
        text = rng.choice(REVIEW_TEMPLATES).format(a=a)
        rows.append({
            "user_idx": u,
            "item_idx": i,
            "review_text": text,
            "rating": rng.randint(1, 5),
            "timestamp": 1_600_000_000 + rng.randint(0, 10_000_000),
        })
    df = pd.DataFrame(rows)
    # Split chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train = df.iloc[: int(0.8 * n)].reset_index(drop=True)
    val = df.iloc[int(0.8 * n) : int(0.9 * n)].reset_index(drop=True)
    test = df.iloc[int(0.9 * n) :].reset_index(drop=True)

    aspect2idx = {a: i for i, a in enumerate(ASPECTS)}
    idx2aspect = {i: a for a, i in aspect2idx.items()}
    vocab = {
        "aspects": ASPECTS,
        "aspect2idx": aspect2idx,
        "idx2aspect": {str(k): v for k, v in idx2aspect.items()},
        "num_aspects": len(ASPECTS),
    }

    # Attach aspect_labels (binary vectors) based on which word appears.
    def label(text):
        v = np.zeros(len(ASPECTS), dtype=np.int8)
        for a in ASPECTS:
            if a in text.lower():
                v[aspect2idx[a]] = 1
        return v

    for d in (train, val, test):
        d["aspect_labels"] = d["review_text"].apply(label)

    return train, val, test, vocab


def _tiny_config():
    from config import Config
    return Config(
        embedding_dim=32,
        num_gcn_layers=2,
        num_aspects=len(ASPECTS),
        num_selected_aspects=3,
        gumbel_tau_start=1.0,
        gumbel_tau_end=0.5,
        dropout=0.1,
        batch_size=16,
        num_epochs=2,
        warmup_epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        lambda_rec=1.0,
        lambda_text=1.0,
        lambda_faith=0.5,
        lambda_sparse=0.01,
        fidelity_margin=0.1,
        patience=10,
        max_explanation_length=32,
        num_neg_samples=20,
        top_k_list=[5, 10],
    )


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
STATE: dict = {}


def stage_1_dataset():
    from data.dataset import (
        InteractionDataset, interaction_collate_fn,
    )
    train = STATE["train"]
    ds = InteractionDataset(train, num_items=NUM_ITEMS)
    loader = DataLoader(ds, batch_size=32, shuffle=True,
                        collate_fn=interaction_collate_fn)
    u, p, n = next(iter(loader))
    assert u.shape == (32,) and p.shape == (32,) and n.shape == (32,)
    assert u.dtype == torch.long
    print(f"  user_ids={tuple(u.shape)}  pos={tuple(p.shape)}  "
          f"neg={tuple(n.shape)}  dtype={u.dtype}")
    return "shapes OK"


def stage_2_lightgcn():
    from models.lightgcn import LightGCN
    train = STATE["train"]
    edge_index = LightGCN.build_edge_index(train, NUM_USERS, NUM_ITEMS)
    expected_edges = 2 * len(train)
    assert edge_index.shape == (2, expected_edges), edge_index.shape

    model = LightGCN(NUM_USERS, NUM_ITEMS, embedding_dim=64, num_layers=3)
    u, i = model(edge_index)
    assert u.shape == (NUM_USERS, 64) and i.shape == (NUM_ITEMS, 64)
    print(f"  edge_index={tuple(edge_index.shape)} "
          f"(expected (2, {expected_edges}))")
    print(f"  user_emb={tuple(u.shape)}  item_emb={tuple(i.shape)}")
    STATE["edge_index"] = edge_index
    STATE["user_emb_graph"] = u.detach()
    STATE["item_emb_graph"] = i.detach()
    return "graph forward OK"


def stage_3_aspect_selector():
    from models.aspect_selector import AspectSelector
    B = 32
    ue = torch.randn(B, 64)
    ie = torch.randn(B, 64)
    sel = AspectSelector(input_dim=128, num_aspects=len(ASPECTS),
                         hidden_dim=64, tau_start=1.0, tau_end=0.5)
    sel.eval()
    with torch.no_grad():
        hard, logits = sel(ue, ie, hard=True)
    assert hard.shape == (B, len(ASPECTS))
    active_per_row = hard.sum(dim=-1).float().mean().item()
    print(f"  selected={tuple(hard.shape)}  logits={tuple(logits.shape)}")
    print(f"  mean active aspects/row (hard) = {active_per_row:.2f}")
    # With random logits and no sparsity pressure, roughly half will be 1.
    # After real training L_sparse pulls this into the 3-5 range.
    assert 0 < active_per_row < len(ASPECTS)
    return f"{active_per_row:.2f} active"


def stage_4_constrained_scorer():
    from models.constrained_scorer import ConstrainedScorer
    B = 32
    ue = torch.randn(B, 64)
    ie = torch.randn(B, 64)
    aspects = torch.rand(B, len(ASPECTS))
    scorer = ConstrainedScorer(user_item_dim=128, aspect_dim=len(ASPECTS),
                               hidden_dim=64)
    with torch.no_grad():
        s = scorer(ue, ie, aspects)
    assert s.shape == (B, 1)
    assert (s >= 0).all() and (s <= 1).all()
    print(f"  score={tuple(s.shape)}  "
          f"range=[{s.min().item():.3f}, {s.max().item():.3f}]")
    return "scores in [0, 1]"


def stage_5_adapter():
    from models.adapter import AspectToT5Adapter
    B = 32
    aspects = torch.rand(B, len(ASPECTS))
    ue = torch.randn(B, 64); ie = torch.randn(B, 64)
    adapter = AspectToT5Adapter(
        aspect_dim=len(ASPECTS), user_item_dim=128,
        t5_hidden_dim=768, num_prefix_tokens=10,
    )
    prefix = adapter(aspects, ue, ie)
    assert prefix.shape == (B, 10, 768)
    print(f"  prefix={tuple(prefix.shape)}  (expected ({B}, 10, 768))")
    return "prefix shape OK"


def stage_6_cafe_rec():
    """End-to-end forward — requires Flan-T5-Base (downloaded/cached)."""
    try:
        from models.cafe_rec import CaFeRec
    except Exception as e:
        raise RuntimeError(f"import failed: {e}")

    cfg = _tiny_config()
    device = _device()
    train = STATE["train"]
    vocab = STATE["vocab"]

    model = CaFeRec(cfg, NUM_USERS, NUM_ITEMS, vocab["aspects"])
    model.to(device)
    model.build_graph(train)
    STATE["model"] = model
    STATE["cfg"] = cfg
    STATE["device"] = device

    u = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.long)
    p = torch.tensor([5, 6, 7, 8], device=device, dtype=torch.long)
    n = torch.tensor([9, 10, 11, 12], device=device, dtype=torch.long)

    exp_ids = model.tokenizer(
        ["Great action movie", "Loved the comedy scenes",
         "Solid drama throughout", "Brilliant thriller direction"],
        padding="max_length", max_length=16, truncation=True,
        return_tensors="pt",
    )["input_ids"].to(device)

    model.train()
    out = model(u, p, n, explanation_ids=exp_ids, mode="train")
    print(f"  rec_score={tuple(out['rec_score'].shape)}  "
          f"neg_rec_score={tuple(out['neg_rec_score'].shape)}")
    print(f"  selected_aspects={tuple(out['selected_aspects'].shape)}  "
          f"aspect_logits={tuple(out['aspect_logits'].shape)}")
    print(f"  explanation_logits="
          f"{tuple(out['explanation_logits'].shape)}  "
          f"explanation_loss={float(out['explanation_loss']):.4f}")
    assert out["rec_score"].shape == (4, 1)
    assert out["selected_aspects"].shape == (4, len(ASPECTS))
    assert out["explanation_logits"].dim() == 3
    return "end-to-end forward OK"


def stage_7_fidelity():
    from losses.fidelity_loss import counterfactual_fidelity_loss
    model = STATE["model"]
    device = STATE["device"]
    model.constrained_scorer.eval()

    # Controlled case: if we pass all-zero aspects to both normal and masked,
    # the two scores are identical → score_drop = 0 → loss should be >= margin.
    B = 4
    emb_dim = STATE["cfg"].embedding_dim
    ue = torch.randn(B, emb_dim, device=device)
    ie = torch.randn(B, emb_dim, device=device)
    zero_aspects = torch.zeros(B, len(ASPECTS), device=device)

    with torch.no_grad():
        s_normal = model.constrained_scorer(ue, ie, zero_aspects).view(-1)
        s_masked = model.constrained_scorer(
            ue, ie, torch.zeros_like(zero_aspects)
        ).view(-1)
    assert torch.allclose(s_normal, s_masked)
    loss_zero, drop_zero = counterfactual_fidelity_loss(
        model.constrained_scorer, ue, ie, zero_aspects, margin=0.1,
    )
    print(f"  [degenerate] drop={float(drop_zero):.4f}  "
          f"loss={float(loss_zero):.4f}  (expect loss ≈ margin=0.1)")
    assert float(drop_zero) < 1e-4
    assert float(loss_zero) > 0.09  # close to the margin of 0.1

    # Non-degenerate case: random soft aspects.
    rand_aspects = torch.rand(B, len(ASPECTS), device=device)
    with torch.no_grad():
        s_normal = model.constrained_scorer(ue, ie, rand_aspects).view(-1)
        s_masked = model.constrained_scorer(
            ue, ie, torch.zeros_like(rand_aspects)
        ).view(-1)
    loss_rand, drop_rand = counterfactual_fidelity_loss(
        model.constrained_scorer, ue, ie, rand_aspects, margin=0.1,
    )
    print(f"  [random]     score_normal≈{float(s_normal.mean()):.3f}  "
          f"score_masked≈{float(s_masked.mean()):.3f}  "
          f"drop={float(drop_rand):.4f}  loss={float(loss_rand):.4f}")
    return f"loss_zero={float(loss_zero):.3f}, loss_rand={float(loss_rand):.3f}"


def stage_8_rec_metrics():
    from evaluation.rec_metrics import recall_at_k, ndcg_at_k
    scores = np.array([0.1, 0.9, 0.5, 0.8, 0.2, 0.7, 0.3, 0.4, 0.6, 0.05])
    gt = 1  # highest score → rank 0
    r10 = recall_at_k(scores, gt, 10)
    n10 = ndcg_at_k(scores, gt, 10)
    r5 = recall_at_k(scores, gt, 5)
    n5 = ndcg_at_k(scores, gt, 5)
    print(f"  gt at rank 0 → Recall@5={r5}  NDCG@5={n5:.4f}  "
          f"Recall@10={r10}  NDCG@10={n10:.4f}")
    assert r10 == 1.0 and r5 == 1.0 and abs(n5 - 1.0) < 1e-6

    # GT far down: rank 9
    scores2 = np.linspace(0.9, 0.1, 10)  # index 0 has highest
    gt2 = 9
    r5_2 = recall_at_k(scores2, gt2, 5)
    r10_2 = recall_at_k(scores2, gt2, 10)
    n10_2 = ndcg_at_k(scores2, gt2, 10)
    print(f"  gt at rank 9 → Recall@5={r5_2}  Recall@10={r10_2}  "
          f"NDCG@10={n10_2:.4f}")
    assert r5_2 == 0.0 and r10_2 == 1.0
    return "recall/ndcg correct"


def stage_9_training_loop():
    """Run 2 tiny epochs; confirm total loss decreases."""
    from data.dataset import ExplanationDataset, explanation_collate_fn
    from losses.bpr_loss import bpr_loss
    from losses.fidelity_loss import counterfactual_fidelity_loss
    from losses.sparsity_loss import sparsity_loss
    from losses.text_loss import text_loss

    model = STATE["model"]; cfg = STATE["cfg"]; device = STATE["device"]
    vocab = STATE["vocab"]; train = STATE["train"]

    ds = ExplanationDataset(
        train, model.tokenizer, vocab["aspect2idx"],
        max_length=cfg.max_explanation_length,
    )
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        collate_fn=explanation_collate_fn,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    epoch_losses: List[float] = []
    for epoch in range(2):
        model.aspect_selector.anneal_temperature(epoch, cfg.num_epochs)
        warmup = epoch < cfg.warmup_epochs
        totals: List[float] = []
        for batch in loader:
            uid = batch["user_idx"].to(device)
            iid = batch["item_idx"].to(device)
            input_ids = batch["input_ids"].to(device)
            neg = torch.randint(
                0, NUM_ITEMS, (uid.size(0),), device=device
            )
            out = model(uid, iid, neg_item_ids=neg,
                        explanation_ids=input_ids, mode="train")
            L_rec = bpr_loss(out["rec_score"], out["neg_rec_score"])
            labels = input_ids.clone()
            labels[labels == model.tokenizer.pad_token_id] = -100
            L_text = text_loss(out["explanation_logits"], labels)
            if warmup:
                L_faith = torch.tensor(0.0, device=device)
                L_sparse = torch.tensor(0.0, device=device)
            else:
                L_faith, _ = counterfactual_fidelity_loss(
                    model.constrained_scorer,
                    out["user_emb"], out["pos_item_emb"],
                    out["selected_aspects"], margin=cfg.fidelity_margin,
                )
                L_sparse = sparsity_loss(out["aspect_logits"])
            total = (
                cfg.lambda_rec * L_rec + cfg.lambda_text * L_text
                + cfg.lambda_faith * L_faith + cfg.lambda_sparse * L_sparse
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            totals.append(float(total.item()))
        avg = float(np.mean(totals))
        epoch_losses.append(avg)
        print(f"  epoch {epoch}  total_loss={avg:.4f}  "
              f"(warmup={warmup}, tau={model.aspect_selector.tau:.3f})")

    # Accept either a decrease, or at least that loss is finite and similar —
    # with 100 users and 2 epochs the signal is noisy, so we assert
    # monotonicity loosely rather than strictly.
    assert all(np.isfinite(epoch_losses))
    if epoch_losses[-1] > epoch_losses[0] * 1.1:
        raise AssertionError(
            f"loss grew meaningfully: {epoch_losses[0]:.4f} -> "
            f"{epoch_losses[-1]:.4f}"
        )
    return f"losses: {[round(x, 4) for x in epoch_losses]}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    t0 = time.time()
    print("=" * 70)
    print("CaFE-Rec pipeline smoke test")
    print("=" * 70)

    train, val, test, vocab = _build_fake_data(seed=0)
    STATE["train"] = train; STATE["val"] = val; STATE["test"] = test
    STATE["vocab"] = vocab
    print(f"[data] train={len(train)}  val={len(val)}  test={len(test)}  "
          f"users={NUM_USERS}  items={NUM_ITEMS}  aspects={len(ASPECTS)}")

    stages = [
        ("1. data/dataset.py", stage_1_dataset),
        ("2. models/lightgcn.py", stage_2_lightgcn),
        ("3. models/aspect_selector.py", stage_3_aspect_selector),
        ("4. models/constrained_scorer.py", stage_4_constrained_scorer),
        ("5. models/adapter.py", stage_5_adapter),
        ("6. models/cafe_rec.py", stage_6_cafe_rec),
        ("7. losses/fidelity_loss.py", stage_7_fidelity),
        ("8. evaluation/rec_metrics.py", stage_8_rec_metrics),
        ("9. full training loop (2 epochs)", stage_9_training_loop),
    ]

    all_passed = True
    for name, fn in stages:
        ok = _run(name, fn)
        if not ok:
            all_passed = False
            # Stages 7 and 9 depend on a working model from stage 6;
            # skip the rest to save noise if the model failed to load.
            if name.startswith("6."):
                print("\n[abort] skipping remaining stages because stage 6 "
                      "(CaFeRec) failed — fix model loading first.")
                break

    # Summary
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for r in _RESULTS:
        mark = "✓" if r.passed else "✗"
        print(f"  {mark} {r.name:<36}  {r.duration:>6.2f}s"
              + (f"   {r.detail}" if r.passed and r.detail else ""))
    print(f"\nTotal time: {elapsed:.2f}s")

    if all_passed:
        print("\n🟢 Pipeline test PASSED")
        return 0
    print("\n🔴 Pipeline test FAILED — see traces above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
