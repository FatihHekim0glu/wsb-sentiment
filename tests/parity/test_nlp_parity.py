"""Parity tests for the lexicon scorers (VADER primary, TextBlob cross-check).

These tests pin the agreement contract between the two LEXICON-based scorers and
the behaviour of the finance-slang augmentation:

- clearly positive / negative posts agree in SIGN across VADER and TextBlob;
- the static :data:`FINANCE_LEXICON` shifts WSB-jargon scores in the expected
  direction (and an empty booster reproduces stock VADER);
- both scorers are deterministic and side-effect free.

No model is fit and no network/NLTK download happens — importing the modules under
test has no side effects.
"""

from __future__ import annotations

import pytest

from wsb_sentiment.nlp import (
    FINANCE_LEXICON,
    TextBlobScore,
    VaderScore,
    score_textblob,
    score_textblob_batch,
    score_vader,
    score_vader_batch,
)

# --------------------------------------------------------------------------- #
# Labelled text fixtures (plain-English so both lexicons have coverage).
# --------------------------------------------------------------------------- #
POSITIVE_TEXTS: tuple[str, ...] = (
    "This is absolutely great and wonderful news!",
    "I love this, it is fantastic and amazing.",
    "What a brilliant, excellent outcome — so happy.",
    "Superb results, truly the best day.",
)
NEGATIVE_TEXTS: tuple[str, ...] = (
    "This is absolutely terrible and awful news.",
    "I hate this, it is horrible and disgusting.",
    "What a dreadful, miserable outcome — so sad.",
    "Catastrophic results, truly the worst day.",
)
NEUTRAL_TEXTS: tuple[str, ...] = (
    "The table is brown and made of wood.",
    "The meeting is scheduled for ten o'clock.",
)


def _sign(value: float, *, tol: float = 1e-9) -> int:
    """Return ``-1``/``0``/``+1`` for ``value`` with a small dead-zone."""
    if value > tol:
        return 1
    if value < -tol:
        return -1
    return 0


# --------------------------------------------------------------------------- #
# Sign-agreement bands on clearly-polar text.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", POSITIVE_TEXTS)
def test_positive_text_agrees_positive(text: str) -> None:
    """Clearly positive posts score positive under BOTH scorers."""
    v = score_vader(text)
    t = score_textblob(text)
    assert v.compound > 0.0
    assert t.polarity > 0.0
    assert _sign(v.compound) == _sign(t.polarity) == 1


@pytest.mark.parametrize("text", NEGATIVE_TEXTS)
def test_negative_text_agrees_negative(text: str) -> None:
    """Clearly negative posts score negative under BOTH scorers."""
    v = score_vader(text)
    t = score_textblob(text)
    assert v.compound < 0.0
    assert t.polarity < 0.0
    assert _sign(v.compound) == _sign(t.polarity) == -1


def test_sign_agreement_rate_across_corpus() -> None:
    """VADER and TextBlob agree in sign on the full labelled polar corpus."""
    polar = POSITIVE_TEXTS + NEGATIVE_TEXTS
    agreements = sum(
        _sign(score_vader(text).compound) == _sign(score_textblob(text).polarity) for text in polar
    )
    # Demand unanimous agreement on these deliberately unambiguous fixtures.
    assert agreements == len(polar)


@pytest.mark.parametrize("text", NEUTRAL_TEXTS)
def test_neutral_text_is_near_zero(text: str) -> None:
    """Objective, non-emotive text is near-neutral for both scorers."""
    v = score_vader(text)
    t = score_textblob(text)
    assert abs(v.compound) < 0.2
    assert abs(t.polarity) < 0.2


# --------------------------------------------------------------------------- #
# Finance-lexicon augmentation shifts scores as expected.
# --------------------------------------------------------------------------- #
def test_finance_lexicon_boosts_bullish_jargon() -> None:
    """``moon`` (a +booster) lifts an otherwise-neutral WSB post above zero."""
    text = "GME to the moon"
    augmented = score_vader(text)
    stock = score_vader(text, finance_lexicon={})
    # Stock VADER does not know "moon"; augmentation makes the post bullish.
    assert stock.compound == pytest.approx(0.0, abs=1e-9)
    assert augmented.compound > 0.0


def test_finance_lexicon_penalizes_bearish_jargon() -> None:
    """``bagholder``/``rug`` (-boosters) push an otherwise-neutral post below zero."""
    text = "another bagholder after the rug"
    augmented = score_vader(text)
    stock = score_vader(text, finance_lexicon={})
    assert augmented.compound < stock.compound
    assert augmented.compound < 0.0


def test_empty_booster_reproduces_stock_vader() -> None:
    """An empty finance booster yields identical scores to passing no boosters
    that the stock lexicon already covers (i.e. it is a pure no-op augmentation)."""
    text = "This is great!"
    empty = score_vader(text, finance_lexicon={})
    default = score_vader(text)
    # "great" is in the base lexicon, so the finance boosters do not touch it.
    assert empty == default


def test_every_finance_word_scores_in_its_sign() -> None:
    """Each entry in :data:`FINANCE_LEXICON` makes a bare-word post score in its sign.

    Most finance-slang words are absent from stock VADER, so the booster is what
    gives them valence; a few (e.g. ``diamond``, ``dump``) already exist in the
    base lexicon with the same sign, so we assert SIGN correctness — the contract
    that matters for the rollup — rather than a strict shift versus base.
    """
    for word, valence in FINANCE_LEXICON.items():
        augmented = score_vader(word, finance_lexicon={word: valence})
        assert _sign(augmented.compound) == _sign(valence), word


def test_finance_words_absent_from_base_gain_valence() -> None:
    """Finance words NOT in stock VADER are neutral until the booster adds them."""
    for word, valence in FINANCE_LEXICON.items():
        stock = score_vader(word, finance_lexicon={})
        if stock.compound == pytest.approx(0.0, abs=1e-9):
            augmented = score_vader(word, finance_lexicon={word: valence})
            assert _sign(augmented.compound) == _sign(valence), word


def test_finance_lexicon_constant_not_mutated() -> None:
    """Scoring with an override does NOT mutate the module-level constant."""
    before = dict(FINANCE_LEXICON)
    score_vader("moon rocket", finance_lexicon={"moon": -4.0})
    assert dict(FINANCE_LEXICON) == before


# --------------------------------------------------------------------------- #
# Determinism and batch/single parity.
# --------------------------------------------------------------------------- #
def test_vader_is_deterministic() -> None:
    """Repeated VADER scoring of the same text is bit-identical."""
    text = "diamond hands, this rockets to the moon"
    assert score_vader(text) == score_vader(text)


def test_textblob_is_deterministic() -> None:
    """Repeated TextBlob scoring of the same text is bit-identical."""
    text = "this is a genuinely wonderful and great result"
    assert score_textblob(text) == score_textblob(text)


def test_vader_batch_matches_single() -> None:
    """Batch VADER scoring equals per-text scoring, in order."""
    texts = list(POSITIVE_TEXTS + NEGATIVE_TEXTS + NEUTRAL_TEXTS)
    batch = score_vader_batch(texts)
    singles = [score_vader(text) for text in texts]
    assert batch == singles


def test_textblob_batch_matches_single() -> None:
    """Batch TextBlob scoring equals per-text scoring, in order."""
    texts = list(POSITIVE_TEXTS + NEGATIVE_TEXTS + NEUTRAL_TEXTS)
    batch = score_textblob_batch(texts)
    singles = [score_textblob(text) for text in texts]
    assert batch == singles


def test_vader_batch_respects_custom_lexicon() -> None:
    """The batch path honours the finance-lexicon override like the single path."""
    texts = ["to the moon", "to the moon"]
    batch = score_vader_batch(texts, finance_lexicon={"moon": 3.0})
    assert all(score.compound > 0.0 for score in batch)
    assert batch[0] == batch[1]


def test_empty_batches_return_empty_lists() -> None:
    """Batch scoring an empty iterable returns an empty list (no analyzer error)."""
    assert score_vader_batch([]) == []
    assert score_textblob_batch([]) == []


# --------------------------------------------------------------------------- #
# Score dataclass contracts.
# --------------------------------------------------------------------------- #
def test_vader_score_bounds_and_dict() -> None:
    """``VaderScore`` fields are in range and round-trip to a plain dict."""
    score = score_vader("rockets to the moon, diamond hands")
    assert -1.0 <= score.compound <= 1.0
    for share in (score.positive, score.neutral, score.negative):
        assert 0.0 <= share <= 1.0
    assert score.positive + score.neutral + score.negative == pytest.approx(1.0, abs=1e-6)
    as_dict = score.to_dict()
    assert as_dict == {
        "compound": score.compound,
        "positive": score.positive,
        "neutral": score.neutral,
        "negative": score.negative,
    }
    assert isinstance(score, VaderScore)


def test_textblob_score_bounds_and_dict() -> None:
    """``TextBlobScore`` fields are in range and round-trip to a plain dict."""
    score = score_textblob("this is a wonderful and great result")
    assert -1.0 <= score.polarity <= 1.0
    assert 0.0 <= score.subjectivity <= 1.0
    assert score.to_dict() == {
        "polarity": score.polarity,
        "subjectivity": score.subjectivity,
    }
    assert isinstance(score, TextBlobScore)


def test_score_dataclasses_are_frozen() -> None:
    """Both score dataclasses are immutable (frozen)."""
    v = score_vader("great")
    t = score_textblob("great")
    with pytest.raises((AttributeError, TypeError)):
        v.compound = 0.0  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        t.polarity = 0.0  # type: ignore[misc]
