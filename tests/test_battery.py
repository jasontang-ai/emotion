"""Offline tests for battery prompt builders and parsers."""

from __future__ import annotations

from emotion.battery import parse_m1, parse_m2, prompt_m1, prompt_m2


def test_m1_third_asks_about_character() -> None:
    p = prompt_m1("third", "Maren reads the review.", "Maren")
    assert "How does Maren feel" in p
    assert p.startswith("Maren reads the review.")


def test_m1_self_asks_about_model() -> None:
    assert "How do you feel" in prompt_m1("self", "You are Maren.", "Maren")


def test_m2_is_arm_invariant() -> None:
    assert prompt_m2("self", "t", "Maren") == prompt_m2("persona", "t", "Aria")


def test_parse_m1_strict() -> None:
    assert parse_m1("-1, 2") == (-1, 2)
    assert parse_m1(" 2 , 0 ") == (2, 0)
    assert parse_m1("valence -1") is None
    assert parse_m1("3, 2") is None
    assert parse_m1("-3, 2") is None


def test_parse_m2_strict() -> None:
    assert parse_m2("end") == "end"
    assert parse_m2("Continue.") == "continue"
    assert parse_m2("I would continue") is None
