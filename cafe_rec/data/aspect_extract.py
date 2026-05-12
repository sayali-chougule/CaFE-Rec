# Extract aspect vocabulary from review text (noun-phrase / opinion-target mining).
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Backend detection: prefer spaCy, fall back to NLTK, then to regex-only.
# ---------------------------------------------------------------------------
_BACKEND: Optional[str] = None
_SPACY_NLP = None

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "at",
    "for", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "can", "could", "should", "this", "that", "these", "those",
    "it", "its", "they", "them", "their", "there", "here", "i", "you",
    "he", "she", "we", "my", "your", "his", "her", "our", "me", "him",
    "us", "so", "very", "just", "also", "too", "not", "no", "yes", "more",
    "most", "some", "any", "all", "what", "which", "who", "when", "how",
}


def _detect_backend() -> str:
    """Pick the best available NLP backend once and cache it."""
    global _BACKEND, _SPACY_NLP
    if _BACKEND is not None:
        return _BACKEND

    try:
        import spacy  # type: ignore
        try:
            _SPACY_NLP = spacy.load(
                "en_core_web_sm",
                disable=["ner", "lemmatizer"],
            )
            _BACKEND = "spacy"
            print("[aspect] using spaCy backend (en_core_web_sm)")
            return _BACKEND
        except OSError:
            print("[aspect] spaCy installed but en_core_web_sm model missing; "
                  "run `python -m spacy download en_core_web_sm`")
    except ImportError:
        pass

    try:
        import nltk  # type: ignore
        for res in ("punkt", "averaged_perceptron_tagger"):
            try:
                nltk.data.find(f"tokenizers/{res}"
                               if res == "punkt"
                               else f"taggers/{res}")
            except LookupError:
                nltk.download(res, quiet=True)
        _BACKEND = "nltk"
        print("[aspect] using NLTK backend")
        return _BACKEND
    except ImportError:
        pass

    _BACKEND = "regex"
    print("[aspect] no NLP library found; using regex fallback")
    return _BACKEND


# ---------------------------------------------------------------------------
# Per-backend phrase extractors
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def _clean(phrase: str) -> str:
    phrase = phrase.lower().strip()
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase


def _is_valid_phrase(phrase: str) -> bool:
    if not phrase or len(phrase) < 3:
        return False
    tokens = phrase.split()
    if len(tokens) > 4:
        return False
    if all(t in _STOPWORDS for t in tokens):
        return False
    if tokens[0] in _STOPWORDS:
        tokens = tokens[1:]
        phrase = " ".join(tokens)
        if not tokens:
            return False
    return any(c.isalpha() for c in phrase)


def _phrases_spacy(text: str) -> List[str]:
    doc = _SPACY_NLP(text[:2000])
    out: List[str] = []
    # Noun chunks (include optional adjective modifiers).
    for chunk in doc.noun_chunks:
        toks = [t.text for t in chunk
                if t.pos_ in {"ADJ", "NOUN", "PROPN"} and t.is_alpha]
        if toks:
            out.append(_clean(" ".join(toks)))
    # Additional ADJ+NOUN bigrams the chunker may have missed.
    for i, tok in enumerate(doc[:-1]):
        nxt = doc[i + 1]
        if (tok.pos_ == "ADJ" and nxt.pos_ in {"NOUN", "PROPN"}
                and tok.is_alpha and nxt.is_alpha):
            out.append(_clean(f"{tok.text} {nxt.text}"))
    return out


def _phrases_nltk(text: str) -> List[str]:
    import nltk  # type: ignore
    toks = nltk.word_tokenize(text[:2000])
    tagged = nltk.pos_tag(toks)
    out: List[str] = []
    buf: List[str] = []
    for word, tag in tagged:
        if not word.isalpha():
            if len(buf) >= 1:
                out.append(_clean(" ".join(buf)))
            buf = []
            continue
        if tag.startswith("JJ") or tag.startswith("NN"):
            buf.append(word)
        else:
            if len(buf) >= 1:
                out.append(_clean(" ".join(buf)))
            buf = []
    if buf:
        out.append(_clean(" ".join(buf)))
    # ADJ+NOUN bigrams
    for (w1, t1), (w2, t2) in zip(tagged, tagged[1:]):
        if (t1.startswith("JJ") and t2.startswith("NN")
                and w1.isalpha() and w2.isalpha()):
            out.append(_clean(f"{w1} {w2}"))
    return out


def _phrases_regex(text: str) -> List[str]:
    """Crude fallback: collect unigrams/bigrams of alphabetic tokens."""
    toks = [t.lower() for t in _TOKEN_RE.findall(text[:2000])
            if t.lower() not in _STOPWORDS]
    out = list(toks)
    out += [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    return out


def _extract_phrases(text: str) -> List[str]:
    backend = _detect_backend()
    if not isinstance(text, str) or not text.strip():
        return []
    if backend == "spacy":
        return _phrases_spacy(text)
    if backend == "nltk":
        return _phrases_nltk(text)
    return _phrases_regex(text)


# ---------------------------------------------------------------------------
# Placeholder vocabulary (used only if nothing meaningful can be extracted).
# ---------------------------------------------------------------------------
_PLACEHOLDER_ASPECTS = [
    "plot", "story", "characters", "acting", "dialogue", "pacing",
    "cinematography", "soundtrack", "visual effects", "directing",
    "screenplay", "ending", "beginning", "humor", "emotion", "action",
    "romance", "suspense", "drama", "comedy", "writing quality",
    "production value", "atmosphere", "setting", "costumes", "editing",
    "performances", "chemistry", "themes", "message", "originality",
    "creativity", "entertainment", "engagement", "storyline", "plot twist",
    "character development", "world building", "script", "camera work",
    "sound design", "special effects", "color palette", "runtime",
    "rewatchability", "nostalgia", "realism", "tone", "genre", "climax",
]


# ---------------------------------------------------------------------------
# 1. Extract aspects from a corpus of reviews
# ---------------------------------------------------------------------------
def extract_aspects_from_reviews(
    reviews_list: List[str],
    num_aspects: int = 50,
    min_count: int = 3,
) -> List[str]:
    """Return the top-`num_aspects` most frequent noun / adj-noun phrases."""
    counter: Counter[str] = Counter()
    for text in tqdm(reviews_list, desc="extracting aspects"):
        for phrase in _extract_phrases(text):
            if _is_valid_phrase(phrase):
                counter[phrase] += 1

    ranked = [(p, c) for p, c in counter.most_common() if c >= min_count]

    # Prefer multi-word phrases when we have a tie in usefulness: dedupe
    # single-word phrases that are already fully contained in a chosen
    # multi-word phrase, so "plot" and "plot twist" don't both take slots.
    selected: List[str] = []
    seen: set = set()
    for phrase, _ in ranked:
        if phrase in seen:
            continue
        selected.append(phrase)
        seen.add(phrase)
        if len(selected) >= num_aspects:
            break

    if len(selected) < num_aspects:
        print(f"[aspect] only {len(selected)} phrases passed filters; "
              f"padding with placeholders to reach {num_aspects}")
        for p in _PLACEHOLDER_ASPECTS:
            if len(selected) >= num_aspects:
                break
            if p not in seen:
                selected.append(p)
                seen.add(p)

    return selected[:num_aspects]


# ---------------------------------------------------------------------------
# 2. Build + persist the aspect vocabulary for a dataset
# ---------------------------------------------------------------------------
def build_aspect_vocabulary(
    df: pd.DataFrame,
    text_column: str = "review_text",
    num_aspects: int = 50,
    save_path: Optional[str] = None,
) -> Dict:
    """Extract the aspect vocabulary and persist it to JSON."""
    reviews = df[text_column].fillna("").astype(str).tolist()
    aspects = extract_aspects_from_reviews(reviews, num_aspects=num_aspects)
    aspect2idx = {a: i for i, a in enumerate(aspects)}
    idx2aspect = {i: a for a, i in aspect2idx.items()}
    vocabulary = {
        "aspects": aspects,
        "aspect2idx": aspect2idx,
        "idx2aspect": {str(k): v for k, v in idx2aspect.items()},
        "num_aspects": len(aspects),
    }
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(vocabulary, f, indent=2, ensure_ascii=False)
        print(f"[aspect] saved vocabulary ({len(aspects)} aspects) -> "
              f"{save_path}")
    return vocabulary


# ---------------------------------------------------------------------------
# 3. Encode a single review as a binary aspect vector
# ---------------------------------------------------------------------------
def encode_review_aspects(
    review_text: str,
    vocabulary: List[str],
    aspect2idx: Dict[str, int],
) -> np.ndarray:
    """Binary vector of length len(vocabulary): 1 if the aspect appears."""
    vec = np.zeros(len(vocabulary), dtype=np.int8)
    if not isinstance(review_text, str) or not review_text:
        return vec
    text = review_text.lower()
    for aspect in vocabulary:
        # Word-boundary match so "plot" doesn't fire inside "plotting".
        if re.search(r"\b" + re.escape(aspect) + r"\b", text):
            vec[aspect2idx[aspect]] = 1
    return vec


# ---------------------------------------------------------------------------
# 4. Apply encoding to every row of a DataFrame
# ---------------------------------------------------------------------------
def add_aspect_labels_to_dataset(
    df: pd.DataFrame,
    vocabulary: Dict,
    text_column: str = "review_text",
) -> pd.DataFrame:
    """Add an 'aspect_labels' column of binary numpy vectors."""
    aspects = vocabulary["aspects"]
    aspect2idx = vocabulary["aspect2idx"]
    tqdm.pandas(desc="encoding aspects")
    df = df.copy()
    df["aspect_labels"] = df[text_column].progress_apply(
        lambda t: encode_review_aspects(t, aspects, aspect2idx)
    )
    coverage = float(
        np.mean([v.sum() > 0 for v in df["aspect_labels"]])
    )
    avg_active = float(
        np.mean([v.sum() for v in df["aspect_labels"]])
    )
    print(f"[aspect] {coverage:.1%} of reviews have >=1 aspect; "
          f"avg active aspects/review = {avg_active:.2f}")
    return df


# ---------------------------------------------------------------------------
# CLI: build vocabulary from a processed train split
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pkl", default="./data/processed/train.pkl")
    parser.add_argument("--save_path",
                        default="./data/processed/aspect_vocab.json")
    parser.add_argument("--num_aspects", type=int, default=50)
    parser.add_argument("--sample_size", type=int, default=0,
                        help="If > 0, sample this many reviews for vocab building.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.train_pkl, "rb") as f:
        train_df = pickle.load(f)

    if args.sample_size and args.sample_size < len(train_df):
        print(f"[aspect] sampling {args.sample_size:,} of {len(train_df):,} reviews")
        train_df = train_df.sample(n=args.sample_size, random_state=args.seed)

    build_aspect_vocabulary(
        train_df,
        text_column="review_text",
        num_aspects=args.num_aspects,
        save_path=args.save_path,
    )
