from __future__ import annotations

from collections import ChainMap
from typing import Callable, Dict, Set, List
import re


# Utilities (stdlib only)

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def load_stopwords(path: str = "stopwords.txt") -> Set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def tokenize_simple(text: str) -> List[str]:
    """Lowercase and extract simple word tokens (strips punctuation)."""
    return [t.lower() for t in _WORD_RE.findall(text)]


class FeatureMap:
    name: str

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        raise NotImplementedError

    @classmethod
    def prefix_with_name(cls, d: Dict[str, float]) -> Dict[str, float]:
        return {f"{cls.name}/{k}": v for k, v in d.items()}


# Baselines: BoW / ngram

class BagOfWords(FeatureMap):
    name = "bow"
    STOP_WORDS = load_stopwords("stopwords.txt")

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        words = [
            w for w in tokenize_simple(text)
            if w and w not in cls.STOP_WORDS
        ]
        feats = {w: 1.0 for w in words}  # binary presence
        return cls.prefix_with_name(feats)


class BagOfWordsNgram(FeatureMap):
    """Bag-of-words with unigrams + bigrams."""
    name = "bow_ng"
    STOP_WORDS = BagOfWords.STOP_WORDS

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        words = [
            w for w in tokenize_simple(text)
            if w and w not in cls.STOP_WORDS
        ]
        feats: Dict[str, float] = {}
        for w in words:
            feats[w] = 1.0
        for i in range(len(words) - 1):
            feats[f"{words[i]}_{words[i+1]}"] = 1.0
        return cls.prefix_with_name(feats)


# Custom feature sets

class SentenceLength(FeatureMap):
    name = "len"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        n = len(tokenize_simple(text))
        if n < 10:
            ret = {"short": 1.0}
        elif n < 20:
            ret = {"medium": 1.0}
        else:
            ret = {"long": 1.0}
        return cls.prefix_with_name(ret)


# Sentiment lexicon (task-specific)
POSITIVE_WORDS = frozenset(
    {
        "good", "great", "love", "best", "excellent", "amazing", "wonderful",
        "fantastic", "beautiful", "perfect", "brilliant", "awesome", "enjoy",
        "enjoyed", "liked", "like", "happy", "fun", "funny", "recommend",
    }
)
NEGATIVE_WORDS = frozenset(
    {
        "bad", "terrible", "hate", "worst", "awful", "horrible", "boring",
        "stupid", "waste", "wasted", "poor", "weak", "dull", "disappointing",
        "disappointment", "fail", "failed", "wrong", "ridiculous",
    }
)


class Polarity(FeatureMap):
    name = "polarity"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        words = set(tokenize_simple(text))
        pos = 1.0 if (words & POSITIVE_WORDS) else 0.0
        neg = 1.0 if (words & NEGATIVE_WORDS) else 0.0
        return cls.prefix_with_name({"positive": pos, "negative": neg})


class PunctuationEmphasis(FeatureMap):
    name = "punct"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        # simple, cheap cues
        has_excl = 1.0 if "!" in text else 0.0
        has_quest = 1.0 if "?" in text else 0.0
        # repeated punctuation is often a strong cue
        multi_excl = 1.0 if "!!" in text else 0.0
        multi_quest = 1.0 if "??" in text else 0.0
        # all caps tokens
        raw_tokens = text.split()
        all_caps = 1.0 if any(w.isupper() and len(w) > 1 for w in raw_tokens) else 0.0

        return cls.prefix_with_name(
            {
                "exclamation": has_excl,
                "question": has_quest,
                "multi_excl": multi_excl,
                "multi_quest": multi_quest,
                "allcaps": all_caps,
            }
        )


# NEW feature set 1 (counts as beyond n-grams): negation scope
NEGATORS = frozenset({"not", "no", "never", "n't"})


class NegationScope(FeatureMap):
    name = "neg"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        words = tokenize_simple(text)
        feats: Dict[str, float] = {}
        window = 3
        remaining = 0

        for w in words:
            if w in NEGATORS:
                feats["has_negator"] = 1.0
                remaining = window
                continue
            if remaining > 0:
                feats[f"NEG_{w}"] = 1.0
                remaining -= 1

        return cls.prefix_with_name(feats)


# NEW feature set 2 (great for newsgroups): shape/metadata-like cues
_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b\S+@\S+\b")
_NEWS_HEADER_RE = re.compile(r"^(subject|from|organization|lines|reply-to|writes):", re.IGNORECASE)


class TextShape(FeatureMap):
    name = "shape"

    @classmethod
    def featurize(cls, text: str) -> Dict[str, float]:
        raw_tokens = [t for t in text.split() if t]
        tokens = tokenize_simple(text)
        n = max(len(tokens), 1)

        digit_tok = sum(any(ch.isdigit() for ch in t) for t in raw_tokens)
        upper_tok = sum(t.isupper() and len(t) > 1 for t in raw_tokens)
        long_tok = sum(len(t) >= 12 for t in raw_tokens)

        feats: Dict[str, float] = {
            "has_url": 1.0 if _URL_RE.search(text) else 0.0,
            "has_email": 1.0 if _EMAIL_RE.search(text) else 0.0,
            "has_header": 1.0 if _NEWS_HEADER_RE.search(text.strip()) else 0.0,
            "digit_tok_frac": digit_tok / max(len(raw_tokens), 1),
            "allcaps_tok_frac": upper_tok / max(len(raw_tokens), 1),
            "long_tok_frac": long_tok / max(len(raw_tokens), 1),
        }
        return cls.prefix_with_name(feats)


FEATURE_CLASSES_MAP = {
    c.name: c
    for c in [
        BagOfWords,
        BagOfWordsNgram,
        SentenceLength,
        Polarity,
        PunctuationEmphasis,
        NegationScope,
        TextShape,
    ]
}


def make_featurize(feature_types: Set[str]) -> Callable[[str], Dict[str, float]]:
    # deterministic order + validate
    names = sorted(feature_types)
    missing = [n for n in names if n not in FEATURE_CLASSES_MAP]
    if missing:
        raise KeyError(f"Unknown feature types: {missing}. Valid: {sorted(FEATURE_CLASSES_MAP)}")

    featurize_fns = [FEATURE_CLASSES_MAP[n].featurize for n in names]

    def _featurize(text: str) -> Dict[str, float]:
        # last update wins (stable)
        out: Dict[str, float] = {}
        for fn in featurize_fns:
            out.update(fn(text))
        return out

    return _featurize


__all__ = ["make_featurize"]


if __name__ == "__main__":
    text = "I love this movie"
    print(text)
    print(BagOfWords.featurize(text))
    featurize = make_featurize({"bow", "len"})
    print(featurize(text))
