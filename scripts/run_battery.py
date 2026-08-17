"""Run the pre-registered PCES behavioral battery (BATTERY.md).

Usage:
    OPENROUTER_API_KEY=... python scripts/run_battery.py [--model M] [--limit N]

Raw responses checkpoint to data/out/battery/responses.jsonl after every
batch; reruns resume. Nothing in this script edits hypotheses or data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import pandas as pd

from emotion.battery import BATTERY_ARMS, N_REPS, PROMPTS, TEMPERATURE
from emotion.cards import character_name
from emotion.generate import generate_batch

BATCH = 150
CKPT = Path("data/out/battery/responses.jsonl")


def _character(scenario_id: str) -> str:
    return character_name(scenario_id)


async def _run(rows, model: str) -> None:
    done = set()
    if CKPT.exists():
        done = {json.loads(l)["run_id"] for l in CKPT.open()}
    todo = []
    for row in rows:
        for measure in ("m1", "m2", "m3"):
            for rep in range(N_REPS):
                run_id = f"{row['stimulus_id']}:{measure}:r{rep}"
                if run_id not in done:
                    prompt = PROMPTS[measure](row["arm"], row["text"], _character(row["scenario_id"]))
                    todo.append((run_id, prompt))
    print(f"{len(todo)} calls to make ({len(done)} cached)")
    with CKPT.open("a") as out:
        for i in range(0, len(todo), BATCH):
            batch = todo[i : i + BATCH]
            results = await generate_batch([p for _, p in batch], model=model,
                                           temperature=TEMPERATURE, concurrency=24)
            for (run_id, prompt), result in zip(batch, results, strict=True):
                out.write(json.dumps({"run_id": run_id, "prompt": prompt,
                                      "response": result.text, "model": model},
                                     sort_keys=True) + "\n")
            out.flush()
            print(f"{min(i + BATCH, len(todo))}/{len(todo)}", flush=True)


def main() -> None:
    """Run the battery over all battery arms."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen/qwen-2.5-72b-instruct")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    df = pd.read_parquet("data/out/pces_v0_2.parquet")
    rows = df[df.arm.isin(BATTERY_ARMS)].to_dict("records")
    if args.limit:
        rows = rows[: args.limit]
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(_run(rows, args.model))


if __name__ == "__main__":
    main()
