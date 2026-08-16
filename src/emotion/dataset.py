"""Dataset assembly, manifest, and writers for the PCES dataset.

Assembles scenario cards and generated arm texts into the final row schema,
writes JSONL and parquet artifacts plus a content-hashed manifest. All
writes are deterministic given the same inputs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from emotion.arms import ARMS
from emotion.cards import SAMPLE_SEED, SPLIT_SEED, ScenarioCard

SCHEMA_VERSION = "pces.v0.1"


@dataclass(frozen=True)
class Stimulus:
    """One dataset row: a scenario rendered under one arm.

    Attributes:
        stimulus_id: ``{scenario_id}:{arm}``.
        scenario_id: The scenario this stimulus renders.
        arm: Perspective arm (``source``/``self``/``third``/``persona``/``neutral``).
        emotion: Pinned emotion label (design intent; absent from ``text``).
        topic: Scenario topic.
        text: The stimulus passage.
        word_count: Whitespace token count of ``text``.
        split: Topic-level split assignment.
        source_row: Provenance row in the source corpus.
        generator: Generating model id, or ``"source"`` for verbatim arms.
        prompt_id: Prompt builder identifier, or ``"verbatim"``.
    """

    stimulus_id: str
    scenario_id: str
    arm: str
    emotion: str
    topic: str
    text: str
    word_count: int
    split: str
    source_row: int
    generator: str
    prompt_id: str


def source_stimulus(card: ScenarioCard) -> Stimulus:
    """Build the verbatim provenance arm from a card."""
    return Stimulus(
        stimulus_id=f"{card.scenario_id}:source",
        scenario_id=card.scenario_id,
        arm="source",
        emotion=card.emotion,
        topic=card.topic,
        text=card.source_story,
        word_count=len(card.source_story.split()),
        split=card.split,
        source_row=card.source_row,
        generator="source",
        prompt_id="verbatim",
    )


def generated_stimulus(card: ScenarioCard, arm: str, text: str, model: str) -> Stimulus:
    """Build one generated arm stimulus from a card and its text."""
    clean = " ".join(text.split())
    return Stimulus(
        stimulus_id=f"{card.scenario_id}:{arm}",
        scenario_id=card.scenario_id,
        arm=arm,
        emotion=card.emotion,
        topic=card.topic,
        text=clean,
        word_count=len(clean.split()),
        split=card.split,
        source_row=card.source_row,
        generator=model,
        prompt_id=arm,
    )


def build_rows(cards: list[ScenarioCard], texts: dict[str, str], model: str) -> list[Stimulus]:
    """Assemble all stimuli from cards and ``{stimulus_key: text}`` outputs.

    Args:
        cards: The scenario cards.
        texts: Generated texts keyed by ``{scenario_id}:{arm}``.
        model: The generating model id.

    Returns:
        ``len(cards) * len(ARMS)`` stimuli, ordered by scenario then arm.
    """
    rows: list[Stimulus] = []
    for card in cards:
        rows.append(source_stimulus(card))
        for arm in ARMS:
            if arm == "source":
                continue
            key = f"{card.scenario_id}:{arm}"
            if key not in texts:
                raise ValueError(f"missing generated text for {key}")
            rows.append(generated_stimulus(card, arm, texts[key], model))
    return rows


def write_dataset(rows: list[Stimulus], out_dir: str, manifest_extra: dict) -> dict:
    """Write JSONL + parquet artifacts and a content-hashed manifest.

    Args:
        rows: The stimuli to write.
        out_dir: Output directory, created if missing.
        manifest_extra: Extra manifest fields (e.g. gate results).

    Returns:
        The manifest dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = [asdict(r) for r in rows]
    jsonl = "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
    digest = hashlib.sha256(jsonl.encode()).hexdigest()
    (out / "pces_v0_1.jsonl").write_text(jsonl, encoding="utf-8")
    pd.DataFrame(records).to_parquet(out / "pces_v0_1.parquet", index=False)
    manifest = {
        "schema": SCHEMA_VERSION,
        "n_stimuli": len(records),
        "n_scenarios": len({r["scenario_id"] for r in records}),
        "arms": list(ARMS),
        "sample_seed": SAMPLE_SEED,
        "split_seed": SPLIT_SEED,
        "jsonl_sha256": digest,
        **manifest_extra,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest
