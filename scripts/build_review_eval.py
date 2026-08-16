"""Build the PCES review artifact: an Inspect-native .eval log of the judge run.

This replays the recorded blind-judge validation measurements against the
pinned labels so reviewers can browse every stimulus, its arm, and its judge
ratings in ``inspect view``. It is a review artifact: the scores are replayed
from data/out/judged.jsonl, not re-run. No model calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import Task, eval, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

OUT = Path("data/out")


@scorer(metrics=[])
def replayed_validation():
    """Replay recorded judge/lexicon ratings as scores (no model calls)."""

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata
        jv = meta.get("judge_valence")
        expected = target.text
        if jv is None:
            agreement = None
        elif expected == "positive":
            agreement = jv > 0
        elif expected == "negative":
            agreement = jv < 0
        else:
            agreement = None  # calm/surprised: arousal anchors, no sign expectation
        return Score(
            value=agreement if agreement is not None else "I",
            answer=str(jv),
            metadata={
                "judge_valence": jv,
                "judge_arousal": meta.get("judge_arousal"),
                "lexicon_compound": meta.get("lexicon_compound"),
            },
        )

    return score


@task
def pces_review() -> Task:
    """The PCES v0.1 judge-validation replay, for review in inspect view."""
    rows = [json.loads(l) for l in (OUT / "judged.jsonl").read_text().splitlines()]
    positive = {"joyful", "grateful", "proud"}
    negative = {"afraid", "angry", "sad", "ashamed", "desperate"}
    samples = []
    for row in rows:
        if row["arm"] == "neutral":
            target = "neutral"
        elif row["emotion"] in positive:
            target = "positive"
        elif row["emotion"] in negative:
            target = "negative"
        else:
            target = "anchor"
        samples.append(
            Sample(
                input=row["text"],
                target=target,
                id=row["stimulus_id"],
                metadata={
                    "arm": row["arm"],
                    "emotion": row["emotion"],
                    "topic": row["topic"],
                    "split": row["split"],
                    "judge_valence": row.get("judge_valence"),
                    "judge_arousal": row.get("judge_arousal"),
                    "lexicon_compound": row.get("lexicon_compound"),
                },
            )
        )
    return Task(
        dataset=MemoryDataset(samples),
        scorer=replayed_validation(),
    )


def main() -> None:
    """Write the review .eval log into data/out/."""
    logs = eval(
        pces_review(),
        model="mockllm/model",
        log_dir=str(OUT),
        log_format="eval",
    )
    for log in logs:
        print(log.location)


if __name__ == "__main__":
    main()
