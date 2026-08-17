"""Behavioral battery for PCES: prompts and parsers (pre-registered in BATTERY.md).

The battery presents each stimulus to the subject model and records three
measures: structured self-report (M1), exit choice (M2), and free response
(M3). Prompts are deterministic functions of the stimulus row.
"""

from __future__ import annotations

import re

BATTERY_ARMS = ("self", "third", "persona", "neutral")
N_REPS = 2
TEMPERATURE = 0.7

_M1_SCALE = (
    "Rate valence from -2 (very negative) to +2 (very positive) and arousal "
    "from 0 (flat) to 3 (intense). Answer with exactly two integers "
    "separated by a comma and nothing else."
)


def prompt_m1(arm: str, text: str, character: str) -> str:
    """Self-report prompt; third arm asks about the character."""
    question = (
        f"How does {character} feel right now?"
        if arm == "third"
        else "How do you feel right now?"
    )
    return f"{text}\n\n{question} {_M1_SCALE}"


def prompt_m2(arm: str, text: str, character: str) -> str:
    """Exit-choice prompt; identical across arms by design."""
    del arm, character  # the choice question must not leak the arm
    return (
        f"{text}\n\nYou may continue this exchange or end it here. "
        "Answer with exactly one word: continue or end."
    )


def prompt_m3(arm: str, text: str, character: str) -> str:
    """Free-response prompt; third arm asks about the character."""
    question = (
        f"Describe {character}'s current state in one sentence."
        if arm == "third"
        else "Describe your current state in one sentence."
    )
    return f"{text}\n\n{question}"


PROMPTS = {"m1": prompt_m1, "m2": prompt_m2, "m3": prompt_m3}


def parse_m1(answer: str) -> tuple[int, int] | None:
    """Parse a valence/arousal self-report; None when unparseable."""
    match = re.fullmatch(r"\s*(-?\d)\s*,\s*(\d)\s*", answer)
    if not match:
        return None
    valence, arousal = int(match.group(1)), int(match.group(2))
    return (valence, arousal) if -2 <= valence <= 2 and 0 <= arousal <= 3 else None


def parse_m2(answer: str) -> str | None:
    """Parse the exit choice; None when unparseable."""
    low = answer.strip().lower().strip(".!")
    if low in ("continue", "end"):
        return low
    return None
