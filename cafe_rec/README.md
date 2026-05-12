# CaFE-Rec: Counterfactually Faithful Explanation Recommender

## Overview

CaFE-Rec is an explainable recommender system that produces natural-language
explanations which are **causally tied** to the recommendation itself. Most
explainable recommenders generate plausible-sounding rationales that do not
actually reflect what drove the model's decision — the explanation is a
post-hoc rationalization. CaFE-Rec addresses this with an information
bottleneck (a Gumbel-Softmax aspect gate) that forces both the recommendation
score and the generated text to flow through the same small set of aspects,
plus a novel **counterfactual fidelity loss** that explicitly penalizes
explanations whose removal does not change the recommendation.

The system combines four modules trained end-to-end: a **LightGCN** backbone
for graph-aware user/item embeddings, an **AspectSelector** (2-layer MLP with
Gumbel-Softmax) that picks 3-5 aspects from a vocabulary extracted from
reviews, a **ConstrainedScorer** MLP that recommends using *only* the
selected aspects, and a **Flan-T5-Base** language model that generates text
from those aspects via a learned linear adapter. Training jointly optimizes
BPR ranking, text cross-entropy, counterfactual fidelity, and L1 sparsity
— targeting RecSys 2026.

## Quick Start

```bash
# 1. Clone
git clone <repo-url> cafe_rec && cd cafe_rec

# 2. Install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm    # optional, for aspect extraction

# 3. Download a dataset (Amazon Reviews 2023, Movies & TV, ~1.2GB)
#    Option A: HuggingFace (automatic, recommended)
#    preprocess.py will fetch from McAuley-Lab/Amazon-Reviews-2023 on first run.
#    Option B: manual — save Movies_and_TV.jsonl.gz into ./data/raw/

mkdir -p data/raw data/processed checkpoints logs results

# 4. Preprocess: k-core filtering, id mapping, chronological split
python3 -m data.preprocess --category movies_tv \
    --data_dir ./data/raw --processed_dir ./data/processed

# 5. Build the aspect vocabulary from the training reviews
python3 -m data.aspect_extract \
    --train_pkl ./data/processed/train.pkl \
    --save_path ./data/processed/aspect_vocab.json \
    --num_aspects 50

# 6. Smoke-test the full pipeline (~60s, no real data needed)
python3 test_pipeline.py

# 7. Train
python3 train.py --dataset amazon_movies --run_name first-run
# Quick debug run (2 epochs, tiny subset):
python3 train.py --debug --no_wandb

# 8. Evaluate the best checkpoint against baselines
python3 evaluate.py \
    --checkpoint ./checkpoints/cafe_rec_amazon_movies_best_*.pt \
    --dataset amazon_movies \
    --baselines ./results/baselines \
    --out_dir ./results
```

## Project Structure

```
cafe_rec/
├── config.py                       # single Config dataclass — all hyperparameters
├── train.py                        # joint-loss training loop, wandb, checkpoints
├── evaluate.py                     # metric + baseline comparison + Wilcoxon
├── test_pipeline.py                # 60-second end-to-end synthetic-data smoke test
├── requirements.txt
│
├── data/
│   ├── preprocess.py               # load, k-core, id mapping, chrono split
│   ├── aspect_extract.py           # spaCy / NLTK / regex aspect mining
│   └── dataset.py                  # Interaction / Explanation / Evaluation datasets
│
├── models/
│   ├── lightgcn.py                 # graph encoder (symmetric-normalized propagation)
│   ├── aspect_selector.py          # Gumbel-Sigmoid info bottleneck
│   ├── constrained_scorer.py       # MLP score over (user, item, aspects)
│   ├── adapter.py                  # aspects -> T5 soft-prefix embeddings
│   └── cafe_rec.py                 # end-to-end wrapper tying the 4 modules
│
├── losses/
│   ├── bpr_loss.py                 # L_rec  : BPR
│   ├── text_loss.py                # L_text : seq2seq cross-entropy
│   ├── fidelity_loss.py            # L_faith: counterfactual fidelity (novel)
│   └── sparsity_loss.py            # L_sparse: L1 on selector logits
│
├── evaluation/
│   ├── rec_metrics.py              # Recall@K, NDCG@K
│   ├── text_metrics.py             # BLEU, ROUGE, BERTScore
│   └── fidelity_metrics.py         # sufficiency, comprehensiveness, CF%
│
└── baselines/
    ├── run_peter.py
    ├── run_pepler.py
    └── run_xrec.py
```

## Architecture

```
(user_id, item_id)
       │
       ▼
 ┌─────────────┐      ┌──────────────────┐
 │  LightGCN   │─────▶│  user_emb [64]   │
 │  (K=3 prop) │      │  item_emb [64]   │
 └─────────────┘      └──────┬───────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ AspectSelector  │  2-layer MLP + Gumbel-Sigmoid
                    │  (info bottleneck)│  → picks 3-5 aspects from 50
                    └───────┬─────────┘
                            │  selected_aspects [50], binary-ish
            ┌───────────────┼─────────────────┐
            ▼                                 ▼
 ┌────────────────────┐              ┌──────────────────┐
 │ ConstrainedScorer  │              │  AspectToT5Ad.   │
 │  MLP over (u,i,a)  │              │  linear projector │
 │  → score ∈ [0,1]   │              │  → prefix [10,768]│
 └────────────────────┘              └──────┬───────────┘
                                            │
                                            ▼
                                 ┌───────────────────────┐
                                 │  Flan-T5-Base         │
                                 │  (soft-prefix encoder)│
                                 │  → explanation text   │
                                 └───────────────────────┘
```

**The four training losses**:

| Loss | Symbol | Role |
|------|--------|------|
| BPR | `L_rec` | rank positive items above sampled negatives |
| Text CE | `L_text` | teach Flan-T5 to produce the reference review text from the aspect prefix |
| Counterfactual fidelity | `L_faith` | *novel* — force the selected aspects to be causal for the score |
| L1 sparsity | `L_sparse` | keep the chosen aspect set small (3-5) |

The total loss is
`λ_rec · L_rec + λ_text · L_text + λ_faith · L_faith + λ_sparse · L_sparse`,
with a **warmup phase** of 5 epochs where only `L_rec + L_text` are active
(the aspect selector has to learn something meaningful before the fidelity
hinge makes sense).

## The Novel Contribution: Counterfactual Fidelity Loss

A self-explaining recommender can still be dishonest: the aspect gate might
pick 3 aspects that *sound* like a good reason while the real decision is
coming from user-id priors or popularity bias hidden elsewhere in the model.
If that happens, removing those 3 aspects won't change the recommendation —
the explanation is correlative, not causal.

Counterfactual fidelity tests this directly:

1. Score the (user, item) pair using the aspects the model picked — this is
   the **factual** recommendation.
2. Zero out those aspects and score again — this is the **counterfactual**.
3. Measure `|factual - counterfactual|`.
4. If the drop is below a margin, penalize the model. Otherwise, no penalty.

Formally:
```
L_faith = mean( max(0, margin - |score(a) - score(0)|) )
```

The margin prevents the trivial solution of an arbitrarily-small drop, and
the absolute-value form cares about magnitude of the causal effect, not
sign. At test time we measure the same quantity three ways:
**sufficiency** (explained aspects alone reproduce the score), **comprehensiveness**
(removing them hurts the score), and **CF%** (fraction of cases where
removing them flips the top-K). See [losses/fidelity_loss.py](losses/fidelity_loss.py#L18-L55)
for the prose derivation alongside the code.

## Running on Apple Silicon (M5 Pro)

The project is developed and tested on an M5 Pro with 24 GB unified memory.
Device selection is automatic:

```python
# config.py
DEVICE = "mps" if torch.backends.mps.is_available() else ...
```

MPS notes observed while building this project:
- LightGCN propagation uses only `scatter_add_` (no `torch_scatter` / `torch_sparse`),
  so it works on MPS out of the box.
- BERTScore (`evaluation/text_metrics.py`) silently falls back to CPU — its
  upstream `bert_score` library does not yet support MPS broadly.
- Flan-T5-Base (≈220 M params) fits comfortably in 24 GB alongside LightGCN
  and the aspect/scorer/adapter stack.
- `DataLoader(num_workers=0)` is the default in `train.py`; multi-worker
  loaders on MPS can cause fork-related slowdowns on macOS.

If you see `RuntimeError: MPS backend out of memory`, drop `batch_size` in
`config.py` from 32 to 16 (Flan-T5 decoder memory scales with sequence
length × batch size).

## Datasets

| Dataset | Category | Source | Approx. size after 5-core |
|---------|----------|--------|---------------------------|
| Amazon Reviews 2023 | Movies & TV | [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | ~300 K users, ~60 K items, ~3 M interactions |
| Yelp | Restaurants/Venues | [yelp.com/dataset](https://www.yelp.com/dataset) | ~150 K users, ~60 K items, ~1.5 M interactions |
| TripAdvisor | Hotels | [Li et al., 2014](https://www.cs.cmu.edu/~jiweil/html/hotel-review.html) or HF mirror | ~20 K users, ~15 K items, ~150 K interactions |

Download commands:

```bash
# Amazon (automatic via HuggingFace datasets)
python3 -c "from datasets import load_dataset; \
    load_dataset('McAuley-Lab/Amazon-Reviews-2023', \
        'raw_review_Movies_and_TV', split='full', trust_remote_code=True)"

# Yelp — register at yelp.com/dataset and drop yelp_academic_dataset_review.json
# into ./data/raw/, then adapt data/preprocess.py (YELP loader TBD).

# TripAdvisor — requires the raw JSON dump; place trip_advisor.jsonl in
# ./data/raw/ and adapt the loader analogously.
```

## Baselines

Each baseline ships as a thin wrapper in `baselines/` that produces
`results/baselines/<name>.json` (aggregate metrics) and optionally
`<name>_per_example.npz` (per-row Recall@10 and CF@10 for Wilcoxon tests).

| Script | Model | Paper |
|--------|-------|-------|
| `run_peter.py` | PETER | Li et al., *ACL 2021* |
| `run_pepler.py` | PEPLER | Li et al., *TOIS 2023* |
| `run_xrec.py` | XRec | 2024 |

Also reproduce NRT (Li et al., *SIGIR 2017*) via an external pointer in
`baselines/run_peter.py`'s README stub — we did not re-implement it here.

```bash
# Run baselines (same preprocessed splits as CaFE-Rec)
python3 -m baselines.run_peter  --dataset amazon_movies \
    --processed_dir ./data/processed --out ./results/baselines
python3 -m baselines.run_pepler --dataset amazon_movies \
    --processed_dir ./data/processed --out ./results/baselines
python3 -m baselines.run_xrec   --dataset amazon_movies \
    --processed_dir ./data/processed --out ./results/baselines
```

## Results

> **These cells are placeholders until the full training runs complete.
> `evaluate.py` will write a real version of this table to
> `results/results_table.json` after each run.**

Amazon Reviews 2023 (Movies & TV), 5-core, chronological split:

| Model          | Recall@10 | NDCG@10 | BLEU-4 | Comp. | Suff. | CF% |
|----------------|-----------|---------|--------|-------|-------|-----|
| NRT            |   TBD     |  TBD    |  TBD   |  Low  |  Low  | N/A |
| PETER          |   TBD     |  TBD    |  TBD   |  Low  |  Low  | N/A |
| PEPLER         |   TBD     |  TBD    |  TBD   |  Low  |  Low  | N/A |
| XRec           |   TBD     |  TBD    |  TBD   |  Med  |  Med  | TBD |
| **CaFE-Rec (ours)** | **TBD**  | **TBD** | **TBD** | **HIGH** | **HIGH** | **TBD*** |

\* statistically significant over the strongest baseline (p < 0.05, paired
Wilcoxon signed-rank) — computed automatically by `evaluate.py`.

## Citation

```bibtex
@inproceedings{sagar2026cafe,
  author    = {Priti Sagar},
  title     = {{CaFE-Rec}: Counterfactually Faithful Explanation Recommender},
  booktitle = {Proceedings of the 20th ACM Conference on Recommender Systems
               (RecSys 2026)},
  year      = {2026},
  note      = {Submitted.}
}
```

## License

MIT License. See [LICENSE](./LICENSE) for details.
