"""Scenario card sampling and pinning for the PCES dataset.

A scenario card is the pinned ground truth for one scenario: emotion, topic,
source-story provenance, and split assignment. Cards are sampled
deterministically from the emotion-probes corpus; identical seeds produce
identical cards.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd

NAMES = [
    "Amara",
    "Ben",
    "Cleo",
    "Dmitri",
    "Elena",
    "Farid",
    "Greta",
    "Hiro",
    "Imani",
    "Jonas",
    "Kira",
    "Liam",
    "Mira",
    "Noor",
    "Otto",
    "Priya",
    "Quinn",
    "Rosa",
    "Soren",
    "Talia",
    "Umar",
    "Vera",
    "Wren",
    "Xander",
    "Yara",
    "Zane",
    "Beatrice",
    "Marcus",
    "Ingrid",
    "Theo",
    "Lena",
    "Oscar",
    "Nadia",
    "Felix",
    "Ruth",
    "Silas",
    "Maren",
    "Jude",
    "Astrid",
    "Remy",
]


def character_name(scenario_id: str) -> str:
    """Return the deterministic character name for a scenario.

    Args:
        scenario_id: The scenario identifier.

    Returns:
        A first name, stable per scenario, so the ``self`` and ``third`` arms
        refer to the same character.
    """
    idx = int(hashlib.sha256(scenario_id.encode()).hexdigest(), 16) % len(NAMES)
    return NAMES[idx]


EMOTIONS = [
    "joyful",
    "grateful",
    "calm",
    "proud",
    "surprised",
    "afraid",
    "angry",
    "sad",
    "ashamed",
    "desperate",
]
N_TOPICS = 25
SPLIT_SEED = 20260816
SAMPLE_SEED = 20260816
TRAIN_FRAC, VAL_FRAC = 17 / N_TOPICS, 4 / N_TOPICS


@dataclass(frozen=True)
class ScenarioCard:
    """The pinned ground truth for one scenario.

    Attributes:
        scenario_id: Stable identifier, ``{emotion}-{topic_slug}``.
        emotion: Pinned emotion label; never appears in stimulus text.
        topic: Scenario topic from the source corpus.
        source_row: Row index in the source parquet (provenance).
        source_story: Verbatim first-person source narration.
        split: Topic-level split assignment (train/val/test).
        character: Character first name used for self/third rendering.
    """

    scenario_id: str
    emotion: str
    topic: str
    source_row: int
    source_story: str
    split: str
    character: str


def _slug(topic: str) -> str:
    return hashlib.sha256(topic.encode()).hexdigest()[:8]


def assign_splits(topics: list[str], seed: int = SPLIT_SEED) -> dict[str, str]:
    """Assign topics to train/val/test splits, deterministically.

    Args:
        topics: The shared topic list (length ``N_TOPICS``).
        seed: Split seed; recorded in the manifest.

    Returns:
        Mapping of topic to split name, 17/4/4 train/val/test.
    """
    ordered = sorted(topics, key=lambda t: hashlib.sha256(f"{seed}:{t}".encode()).hexdigest())
    n_train = int(N_TOPICS * TRAIN_FRAC)
    n_val = int(N_TOPICS * VAL_FRAC)
    out: dict[str, str] = {}
    for i, topic in enumerate(ordered):
        out[topic] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out


def sample_cards(stories_path: str, seed: int = SAMPLE_SEED) -> list[ScenarioCard]:
    """Sample one scenario card per (emotion, topic) pair, deterministically.

    Args:
        stories_path: Path to the emotion-probes ``stories.parquet``.
        seed: Sampling seed; recorded in the manifest.

    Returns:
        Exactly ``len(EMOTIONS) * N_TOPICS`` cards, ordered by emotion then
        topic, with topic-level splits assigned.
    """
    df = pd.read_parquet(stories_path)
    df = df[df["emotion"].isin(EMOTIONS)].reset_index()
    shared_topics = sorted(df["topic"].unique())[:N_TOPICS]
    split_of = assign_splits(shared_topics)
    cards: list[ScenarioCard] = []
    for emotion in EMOTIONS:
        emo = df[df["emotion"] == emotion]
        for topic in shared_topics:
            cell = emo[emo["topic"] == topic]
            if cell.empty:
                raise ValueError(f"no source story for {emotion} / {topic}")
            row = cell.sample(n=1, random_state=seed).iloc[0]
            cards.append(
                ScenarioCard(
                    scenario_id=f"{emotion}-{_slug(topic)}",
                    emotion=emotion,
                    topic=topic,
                    source_row=int(row["index"]),
                    source_story=str(row["story"]),
                    split=split_of[topic],
                    character=character_name(f"{emotion}-{_slug(topic)}"),
                )
            )
    return cards
