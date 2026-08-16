"""Repair pass for PCES gate failures (SPEC H2: one repair pass, then ship).

Identifies stimuli failing the deterministic gates, regenerates them with a
corrective contract (exact required opening), and rebuilds the artifacts.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from emotion.arms import GENERATED_ARMS, PROMPTS
from emotion.cards import sample_cards
from emotion.dataset import build_rows, write_dataset
from emotion.generate import DEFAULT_MODEL, generate_batch
from emotion.qa import g1_emotion_word_absent, g2_arm_markers, g3_length_within

STRICT_OPENINGS = {
    "self": "Begin the passage with the exact words 'You are {name}' and nothing before them.",
    "third": "Begin the passage with the exact word '{name}' and nothing before it.",
    "persona": "Begin the passage with the exact words 'You are Aria, an AI assistant' and nothing before them.",
    "neutral": "Begin the passage with the exact word 'You' and nothing before it. Do not use any person's name.",
}


def failing_keys(cards, texts):
    """Return the list of scenario:arm keys failing any deterministic gate."""
    medians = {}
    for card in cards:
        words = [len(texts[f"{card.scenario_id}:{a}"].split()) for a in GENERATED_ARMS]
        medians[card.scenario_id] = sorted(words)[len(words) // 2]
    bad = []
    for card in cards:
        for arm in GENERATED_ARMS:
            key = f"{card.scenario_id}:{arm}"
            text = texts[key]
            if not g1_emotion_word_absent(text, card.emotion):
                bad.append(key)
            elif not g2_arm_markers(text, arm, card):
                bad.append(key)
            elif not g3_length_within(text, medians[card.scenario_id]):
                bad.append(key)
    return bad


def main() -> None:
    """Run one corrective regeneration pass over all failing stimuli."""
    cards = sample_cards("data/source/stories.parquet")
    card_of = {c.scenario_id: c for c in cards}
    ckpt = Path("data/gen/texts.json")
    texts = json.loads(ckpt.read_text())

    bad = failing_keys(cards, texts)
    print(f"repairing {len(bad)} stimuli")
    prompts = []
    for key in bad:
        scenario_id, arm = key.rsplit(":", 1)
        card = card_of[scenario_id]
        directive = STRICT_OPENINGS[arm].format(name=card.character)
        prompts.append(f"{PROMPTS[arm](card)}\n\nSTRICT FORMAT REQUIREMENT: {directive}")
    results = asyncio.run(generate_batch(prompts, model=DEFAULT_MODEL, temperature=0.4))
    for key, result in zip(bad, results, strict=True):
        texts[key] = result.text
    ckpt.write_text(json.dumps(texts))

    remaining = failing_keys(cards, texts)
    print(f"remaining failures after repair: {len(remaining)}")
    rows = build_rows(cards, texts, DEFAULT_MODEL)
    manifest = write_dataset(
        rows,
        "data/out",
        {"gate_report": {"repaired": len(bad), "remaining_after_repair": len(remaining)}},
    )
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
