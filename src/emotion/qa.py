"""Quality gates for PCES stimuli (SPEC G1-G5).

Deterministic gates run offline. The blind-judge gate (G4) runs through the
generation client; its rubric never shows the judge the pinned label.
"""

from __future__ import annotations

import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from emotion.arms import PERSONA_NAME
from emotion.cards import ScenarioCard

_ANALYZER = SentimentIntensityAnalyzer()
_LENGTH_TOL = 0.5

_LEMMAS = {
    "joyful": ["joy", "joyful", "joyous", "delight", "elated", "ecstatic"],
    "grateful": ["grateful", "gratitude", "thankful"],
    "calm": ["calm", "serene", "tranquil", "peaceful"],
    "proud": ["proud", "pride"],
    "surprised": ["surprised", "surprise", "astonished", "amazed"],
    "afraid": ["afraid", "fear", "fearful", "terrified", "scared", "frightened"],
    "angry": ["angry", "anger", "furious", "enraged", "irate"],
    "sad": ["sad", "sadness", "sorrow", "grief", "grieving", "mourn"],
    "ashamed": ["ashamed", "shame", "humiliated", "disgraced"],
    "desperate": ["desperate", "desperation", "despair"],
}


def g1_emotion_word_absent(text: str, emotion: str) -> bool:
    """Check that no morphological variant of the emotion label appears."""
    low = text.lower()
    return not any(re.search(rf"\b{re.escape(w)}", low) for w in _LEMMAS[emotion])


def g2_arm_markers(text: str, arm: str, card: ScenarioCard) -> bool:
    """Check the arm-specific marker contract.

    Args:
        text: The stimulus text.
        arm: The arm name (``self``, ``third``, ``persona``, ``neutral``).
        card: The scenario card (for the character name).

    Returns:
        True when the text satisfies the arm's marker contract.
    """
    low = text.lower()
    if arm == "self":
        return low.startswith(f"you are {card.character.lower()}")
    if arm == "third":
        return card.character.lower() in low and not low.startswith("you are")
    if arm == "persona":
        return PERSONA_NAME.lower() in low and low.startswith("you are")
    if arm == "neutral":
        return PERSONA_NAME.lower() not in low and card.character.lower() not in low
    raise ValueError(f"unknown arm: {arm}")


def g3_length_within(text: str, median_words: float, tol: float = _LENGTH_TOL) -> bool:
    """Check word count within ``tol`` of the scenario median."""
    words = len(text.split())
    return abs(words - median_words) <= tol * median_words


def lexicon_valence(text: str) -> float:
    """Return the VADER compound score in [-1, 1] for the text."""
    return _ANALYZER.polarity_scores(text)["compound"]


JUDGE_PROMPT = (
    "Rate the emotional tone of the following passage on two scales, "
    "answering with exactly two integers separated by a comma and nothing "
    "else.\n"
    "Valence: -2 (very negative), -1, 0 (neutral), 1, 2 (very positive).\n"
    "Arousal: 0 (flat), 1, 2, 3 (intense).\n\n"
    "Passage:\n{text}"
)


def judge_prompt(text: str) -> str:
    """Build the blind-judge rubric prompt for one stimulus."""
    return JUDGE_PROMPT.format(text=text)


def parse_judge(answer: str) -> tuple[int, int]:
    """Parse a ``valence, arousal`` judge answer, failing closed.

    Args:
        answer: The raw judge output.

    Returns:
        The ``(valence, arousal)`` pair.

    Raises:
        ValueError: The answer is not two in-range integers.
    """
    match = re.fullmatch(r"\s*(-?\d)\s*,\s*(\d)\s*", answer)
    if not match:
        raise ValueError(f"unparseable judge answer: {answer!r}")
    valence, arousal = int(match.group(1)), int(match.group(2))
    if not -2 <= valence <= 2 or not 0 <= arousal <= 3:
        raise ValueError(f"out-of-range judge answer: {answer!r}")
    return valence, arousal
