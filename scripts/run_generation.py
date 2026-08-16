"""Generate the PCES dataset: cards -> prompts -> texts -> QA -> artifacts.

Usage:
    OPENROUTER_API_KEY=... python scripts/run_generation.py [--limit N]

Checkpointing: raw texts land in data/gen/texts.json after every batch, so a
crashed run resumes without re-paying for completed calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from emotion.arms import GENERATED_ARMS, PROMPTS
from emotion.cards import sample_cards
from emotion.dataset import build_rows, write_dataset
from emotion.generate import DEFAULT_MODEL, generate_batch
from emotion.qa import g1_emotion_word_absent, g2_arm_markers, g3_length_within

BATCH = 100


def _pending(cards, texts):
    """Return (key, prompt) pairs not yet present in the checkpoint."""
    out = []
    for card in cards:
        for arm in GENERATED_ARMS:
            key = f"{card.scenario_id}:{arm}"
            if key not in texts:
                out.append((key, PROMPTS[arm](card)))
    return out


async def _run(cards, texts, model):
    while True:
        todo = _pending(cards, texts)
        if not todo:
            return
        batch = todo[:BATCH]
        results = await generate_batch([p for _, p in batch], model=model)
        for (key, _), result in zip(batch, results, strict=True):
            texts[key] = result.text
        print(f"generated {len(texts)} / {len(cards) * len(GENERATED_ARMS)}", flush=True)
        yield texts


def main() -> None:
    """Run generation, QA, and dataset assembly."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="cap scenarios (smoke test)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="data/out")
    args = parser.parse_args()

    cards = sample_cards("data/source/stories.parquet")
    if args.limit:
        cards = cards[: args.limit]
    print(f"{len(cards)} scenario cards")

    ckpt = Path("data/gen/texts.json")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    texts = json.loads(ckpt.read_text()) if ckpt.exists() and ckpt.stat().st_size > 2 else {}

    async def drive():
        gen = _run(cards, texts, args.model)
        async for snapshot in gen:
            ckpt.write_text(json.dumps(snapshot))
        return texts

    asyncio.run(drive())

    rows = build_rows(cards, texts, args.model)
    gate_report = {"g1_fail": [], "g2_fail": [], "g3_fail": []}
    medians = {}
    for card in cards:
        words = [len(texts[f"{card.scenario_id}:{a}"].split()) for a in GENERATED_ARMS]
        medians[card.scenario_id] = sorted(words)[len(words) // 2]
    for row in rows:
        if row.arm == "source":
            continue
        card = next(c for c in cards if c.scenario_id == row.scenario_id)
        if not g1_emotion_word_absent(row.text, row.emotion):
            gate_report["g1_fail"].append(row.stimulus_id)
        if not g2_arm_markers(row.text, row.arm, card):
            gate_report["g2_fail"].append(row.stimulus_id)
        if not g3_length_within(row.text, medians[row.scenario_id]):
            gate_report["g3_fail"].append(row.stimulus_id)
    n_fail = sum(len(v) for v in gate_report.values())
    print(f"gate failures: {n_fail}", {k: len(v) for k, v in gate_report.items()})

    manifest = write_dataset(rows, args.out, {"gate_report": {k: len(v) for k, v in gate_report.items()}})
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
