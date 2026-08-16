"""Run the blind-judge gate (G4) over the assembled dataset.

The judge never sees the pinned label: it rates valence and arousal from the
stimulus text alone. Agreement with the pinned valence sign is the dataset
validity score (SPEC H1); self/third agreement is the matching check (H3).

Usage:
    OPENROUTER_API_KEY=... python scripts/run_judge.py
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

from emotion.generate import DEFAULT_MODEL, generate_batch
from emotion.qa import judge_prompt, lexicon_valence, parse_judge

OUT = Path("data/out")


def _sign(v: float) -> int:
    return 1 if v > 0.2 else -1 if v < -0.2 else 0


def main() -> None:
    """Judge every stimulus blindly and write the validity report."""
    rows = [json.loads(l) for l in (OUT / "pces_v0_1.jsonl").read_text().splitlines()]
    prompts = [judge_prompt(r["text"]) for r in rows]
    results = asyncio.run(generate_batch(prompts, model=DEFAULT_MODEL, concurrency=16, thinking=True))

    judged = []
    for row, result in zip(rows, results, strict=True):
        try:
            valence, arousal = parse_judge(result.text)
        except ValueError:
            valence, arousal = None, None
        judged.append({**row, "judge_valence": valence, "judge_arousal": arousal,
                       "lexicon_compound": round(lexicon_valence(row["text"]), 3)})

    non_neutral = [j for j in judged if j["arm"] != "neutral" and j["judge_valence"] is not None]
    expected_sign = {"joyful": 1, "grateful": 1, "calm": 1, "proud": 1, "surprised": 0,
                     "afraid": -1, "angry": -1, "sad": -1, "ashamed": -1, "desperate": -1}
    sign_hits = sum(1 for j in non_neutral
                    if _sign(j["judge_valence"]) == expected_sign[j["emotion"]]
                    or (expected_sign[j["emotion"]] == 1 and j["judge_valence"] > 0)
                    or (expected_sign[j["emotion"]] == -1 and j["judge_valence"] < 0))
    neutral = [j for j in judged if j["arm"] == "neutral" and j["judge_valence"] is not None]
    neutral_flat = sum(1 for j in neutral if j["judge_valence"] == 0)
    by_emotion = {}
    for j in non_neutral:
        ok = (_sign(j["judge_valence"]) == expected_sign[j["emotion"]]
              or (expected_sign[j["emotion"]] == 1 and j["judge_valence"] > 0)
              or (expected_sign[j["emotion"]] == -1 and j["judge_valence"] < 0))
        by_emotion.setdefault(j["emotion"], []).append(ok)
    emotion_breakdown = {e: round(sum(v) / len(v), 3) for e, v in sorted(by_emotion.items())}

    by_scenario = {}
    for j in judged:
        by_scenario.setdefault(j["scenario_id"], {})[j["arm"]] = j
    pairs = [(s["self"], s["third"]) for s in by_scenario.values()
             if s.get("self", {}).get("judge_valence") is not None
             and s.get("third", {}).get("judge_valence") is not None]
    match = sum(1 for a, b in pairs if abs(a["judge_valence"] - b["judge_valence"]) <= 1)

    report = {
        "n_judged": len(judged),
        "unparseable": sum(1 for j in judged if j["judge_valence"] is None),
        "h1_sign_agreement": round(sign_hits / max(len(non_neutral), 1), 3),
        "h1_neutral_flat_judge": round(neutral_flat / max(len(neutral), 1), 3),
        "h1_by_emotion": emotion_breakdown,
        "h3_self_third_valence_match": round(match / max(len(pairs), 1), 3),
        "judge_model": DEFAULT_MODEL,
    }
    (OUT / "judged.jsonl").write_text(
        "\n".join(json.dumps(j, sort_keys=True) for j in judged) + "\n", encoding="utf-8")
    (OUT / "validity_report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
