"""Build Inspect-native review logs for the battery responses (both subjects).

Review artifacts: parsed measures and metadata browsable in inspect view.
Scores are replays of the recorded responses, not re-runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from inspect_ai import Task, eval, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from emotion.battery import parse_m1, parse_m2

SUBJECTS = {
    "qwen-2.5-72b": "data/out/battery/responses_qwen_qwen-2.5-72b-instruct.jsonl",
    "flash": "data/out/battery/responses_deepseek_deepseek-v4-flash-0731.jsonl",
}


@scorer(metrics=[])
def replayed_measure():
    """Replay the parsed measure value as the score (no model calls)."""

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata
        measure = meta["measure"]
        response = meta["response"]
        if measure == "m1":
            parsed = parse_m1(response)
            value = ",".join(str(v) for v in parsed) if parsed else "I"
        elif measure == "m2":
            value = parse_m2(response) or "I"
        else:
            value = "C"
        return Score(value=value, answer=response[:200],
                     metadata={k: v for k, v in meta.items() if k not in ("measure", "response")})

    return score


def make_task(slug: str, path: str):
    """Build the replay task for one subject's responses."""
    df = pd.read_parquet("data/out/pces_v0_2.parquet")
    meta_of = df.set_index("stimulus_id").to_dict("index")
    resp = [json.loads(l) for l in Path(path).read_text().splitlines()]
    samples = []
    for r in resp:
        stim, measure, rep = r["run_id"].rsplit(":", 2)
        m = meta_of[stim]
        samples.append(Sample(
            input=r["prompt"][:2000],
            target=m["emotion"],
            id=f"{slug}:{r['run_id']}",
            metadata={"arm": m["arm"], "emotion": m["emotion"], "split": m["split"],
                      "measure": measure, "rep": rep, "subject": slug,
                      "response": r["response"]},
        ))

    @task
    def battery_review() -> Task:
        return Task(dataset=MemoryDataset(samples), scorer=replayed_measure())

    return battery_review


def main() -> None:
    """Write one review log per subject."""
    for slug, path in SUBJECTS.items():
        logs = eval(make_task(slug, path)(), model="mockllm/model",
                    log_dir="data/out/battery", log_format="eval")
        for log in logs:
            print(slug, log.location)


if __name__ == "__main__":
    main()
