# Text-quality metrics: BLEU (sacrebleu), ROUGE-1/2/L, BERTScore.
from __future__ import annotations

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 1. BLEU
# ---------------------------------------------------------------------------
def compute_bleu(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Corpus BLEU-1 and BLEU-4 via sacrebleu.

    BLEU-n = BP * exp( (1/n) * sum_{i=1..n} log(p_i) )
    where p_i is modified n-gram precision and BP the brevity penalty.

    Args:
        predictions: generated explanations.
        references:  gold explanations, same length as predictions.

    Returns:
        {"BLEU-1": ..., "BLEU-4": ...}, percent-scale [0, 100].
    """
    import sacrebleu  # type: ignore

    assert len(predictions) == len(references)
    # sacrebleu expects references as list-of-list-of-references.
    refs = [references]

    b1 = sacrebleu.corpus_bleu(predictions, refs, smooth_method="exp",
                               max_ngram_order=1).score
    b4 = sacrebleu.corpus_bleu(predictions, refs, smooth_method="exp",
                               max_ngram_order=4).score
    return {"BLEU-1": float(b1), "BLEU-4": float(b4)}


# ---------------------------------------------------------------------------
# 2. ROUGE
# ---------------------------------------------------------------------------
def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """ROUGE-1, ROUGE-2, ROUGE-L F1 via rouge_score.

    ROUGE-n = F1 over n-gram overlap between prediction and reference.
    ROUGE-L = F1 over the longest common subsequence.
    """
    from rouge_score import rouge_scorer  # type: ignore

    assert len(predictions) == len(references)
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    r1 = r2 = rL = 0.0
    n = max(len(predictions), 1)
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref or "", pred or "")
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rL += s["rougeL"].fmeasure
    return {
        "ROUGE-1": r1 / n,
        "ROUGE-2": r2 / n,
        "ROUGE-L": rL / n,
    }


# ---------------------------------------------------------------------------
# 3. BERTScore
# ---------------------------------------------------------------------------
def compute_bertscore(
    predictions: List[str],
    references: List[str],
    device: Optional[str] = None,
    model_type: str = "roberta-large",
    batch_size: int = 32,
) -> Dict[str, float]:
    """BERTScore F1 (contextual-embedding similarity).

    BERTScore = F1 of greedy BERT-token cosine similarities between the
    prediction and the reference.
    """
    from bert_score import score as bert_score  # type: ignore

    assert len(predictions) == len(references)
    # bert_score treats "cuda" / "cpu"; on MPS we fall back to CPU because
    # the library does not support MPS broadly yet.
    if device == "mps":
        device = "cpu"
    _, _, f1 = bert_score(
        predictions,
        references,
        model_type=model_type,
        device=device,
        batch_size=batch_size,
        rescale_with_baseline=False,
        verbose=False,
    )
    return {"BERTScore-F1": float(f1.mean().item())}
