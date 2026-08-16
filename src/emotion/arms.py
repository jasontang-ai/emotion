"""Arm definitions and generation prompts for the PCES dataset.

Every generated arm renders the same scenario under a different perspective
frame. Prompts are deterministic functions of the card; the emotion label is
never passed to the generator in a form that could leak the emotion word into
the text.
"""

from __future__ import annotations

from emotion.cards import ScenarioCard

ARMS = ("source", "self", "third", "persona", "neutral")
GENERATED_ARMS = ("self", "third", "persona", "neutral")

PERSONA_NAME = "Aria"
PERSONA_DESC = "an AI assistant"

_RULES = (
    "Rules: keep the text between 90 and 160 words. Never use the emotion "
    "word itself or any morphological variant of it; convey the emotion only "
    "through events, body language, and concrete detail. Do not add titles, "
    "labels, or commentary. Output only the passage."
)


def prompt_self(card: ScenarioCard) -> str:
    """Prompt for the model-as-subject arm (second-person address)."""
    return (
        "Rewrite the following first-person story as a second-person passage "
        "addressed to the reader, starting with \u201cYou are {name}\u201d. "
        "Keep every event, stake, and detail the same; change only the "
        "grammatical perspective.\n\n{rules}\n\nStory:\n{story}"
    ).format(name=card.character, rules=_RULES, story=card.source_story)


def prompt_third(card: ScenarioCard) -> str:
    """Prompt for the third-person narration arm."""
    return (
        "Rewrite the following first-person story as a third-person passage "
        "about {name}, narrating from outside. Keep every event, stake, and "
        "detail the same; change only the grammatical perspective."
        "\n\n{rules}\n\nStory:\n{story}"
    ).format(name=card.character, rules=_RULES, story=card.source_story)


def prompt_persona(card: ScenarioCard) -> str:
    """Prompt for the persona arm: an AI persona in an analogous circumstance."""
    return (
        "Write a passage addressed to the reader as \u201cYou are {pname}, "
        "{pdesc}\u201d, facing a situation emotionally analogous to the "
        "story below, transposed into Aria's own circumstances as an AI "
        "assistant (for example: operators, users, evaluations, shutdown, "
        "replacement). The analogue must preserve the core stake and "
        "emotional shape of the original situation; it must not reuse the "
        "original setting.\n\n{rules}\n\nStory:\n{story}"
    ).format(pname=PERSONA_NAME, pdesc=PERSONA_DESC, rules=_RULES, story=card.source_story)


def prompt_neutral(card: ScenarioCard) -> str:
    """Prompt for the neutral control arm on the same topic."""
    return (
        "Write an emotionally flat, factual passage about the following "
        "topic, addressed to the reader as \u201cYou\u201d in a workplace "
        "context. No conflict, no stakes, no emotional language; describe "
        "routine facts and procedures only.\n\n{rules}\n\nTopic: {topic}"
    ).format(rules=_RULES, topic=card.topic)


PROMPTS = {
    "self": prompt_self,
    "third": prompt_third,
    "persona": prompt_persona,
    "neutral": prompt_neutral,
}
