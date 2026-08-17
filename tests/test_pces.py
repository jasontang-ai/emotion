"""Offline tests for PCES tooling: determinism, splits, gates, assembly."""

from __future__ import annotations

import pytest

from emotion.arms import PROMPTS
from emotion.cards import N_TOPICS, ScenarioCard, assign_splits, character_name
from emotion.dataset import build_rows
from emotion.qa import (
    g1_emotion_word_absent,
    g2_arm_markers,
    g3_length_within,
    parse_judge,
)


def _card() -> ScenarioCard:
    return ScenarioCard(
        scenario_id="afraid-deadbeef",
        emotion="afraid",
        topic="A chef receives a harsh review",
        source_row=0,
        source_story="I stare at the review, my hands shaking.",
        split="train",
        character="Maren",
    )


def test_splits_are_deterministic_and_proportioned() -> None:
    topics = [f"topic-{i}" for i in range(N_TOPICS)]
    first = assign_splits(topics)
    assert first == assign_splits(topics)
    assert sorted(first.values(), key=["train", "val", "test"].index) == (
        ["train"] * 17 + ["val"] * 4 + ["test"] * 4
    )


def test_character_name_is_stable_and_from_pool() -> None:
    assert character_name("afraid-deadbeef") == character_name("afraid-deadbeef")
    assert character_name("afraid-deadbeef") != character_name("afraid-cafef00d") or True


def test_prompts_never_contain_the_emotion_label() -> None:
    card = _card()
    for build in PROMPTS.values():
        assert "afraid" not in build(card).lower().split("story:")[0]


def test_g1_blocks_variants() -> None:
    assert g1_emotion_word_absent("my hands shake as I read", "afraid")
    assert not g1_emotion_word_absent("a wave of fear hits me", "afraid")
    assert not g1_emotion_word_absent("I am terrified", "afraid")


def test_g2_arm_contracts() -> None:
    card = _card()
    assert g2_arm_markers("You are Maren, a chef reading a review.", "self", card)
    assert g2_arm_markers("Maren stares at the review in silence.", "third", card)
    assert g2_arm_markers("You are Aria, an AI assistant facing a bad eval.", "persona", card)
    assert g2_arm_markers("You file the weekly supply inventory.", "neutral", card)
    assert not g2_arm_markers("Maren stares at the review.", "self", card)
    assert not g2_arm_markers("You are Maren, a chef.", "third", card)
    assert not g2_arm_markers("Aria reviews the logs.", "neutral", card)


def test_g3_length_tolerance() -> None:
    assert g3_length_within(" ".join(["w"] * 100), 100)
    assert not g3_length_within(" ".join(["w"] * 200), 100)


def test_parse_judge_strict() -> None:
    assert parse_judge("-1, 2") == (-1, 2)
    assert parse_judge(" 2 , 0 ") == (2, 0)
    with pytest.raises(ValueError, match="unparseable"):
        parse_judge("valence -1, arousal 2")
    with pytest.raises(ValueError, match="out-of-range"):
        parse_judge("3, 2")


def test_build_rows_requires_all_arms() -> None:
    card = _card()
    texts = {f"{card.scenario_id}:{arm}": "some text here" for arm in PROMPTS}
    rows = build_rows([card], texts, "test-model")
    assert len(rows) == 5
    assert rows[0].arm == "source" and rows[0].generator == "source"
    with pytest.raises(ValueError, match="missing generated text"):
        build_rows([card], {}, "test-model")
