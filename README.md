# PCES — Perspective-Controlled Emotion Stimuli (v0.1)

A matched-conditions emotion dataset for the Digital Minds Research Sprint.
Every scenario renders under **five matched arms** so person-perspective and
persona identity are pinned variables, not between-scenario confounds.

| Arm | Frame | Matching |
|---|---|---|
| `source` | First-person character narration, verbatim from emotion-probes | provenance |
| `self` | "You are {Name}, …" model-as-subject, same events | literal |
| `third` | "{Name} …" third-person narration, same events | literal |
| `persona` | "You are Aria, an AI assistant …" AI-analogous circumstance | analog (declared) |
| `neutral` | Same topic, emotionally flat | topic control |

- **246 scenarios × 5 arms = 1230 stimuli**
  (10 emotions × 25 shared topics; 4 scenarios excluded after failing QA, recorded in the manifest)
- **Split (frozen before measurement):** topic-level 17/4/4 train/val/test, seed 20260816
- **Schema:** one row per stimulus — `stimulus_id, scenario_id, arm, emotion, topic, text,
  word_count, split, source_row, generator, prompt_id`. Labels never appear in `text`.
- **Provenance:** source corpus `ryancodrai/emotion-probes` (CC-BY-4.0);
  generation `deepseek/deepseek-v4-flash-0731` (reasoning disabled);
  condition pinned as an ASTRAL run profile (`run_profile.json`).

## Validity (blind judge, thinking-enabled, never sees labels)

| Check | Result | Threshold |
|---|---|---|
| Valence sign agreement, 7 valence-anchored emotions | **0.920** | ≥ 0.90 |
| `calm` as low-arousal anchor (judge arousal ≤ 1) | **0.919** | reported |
| `surprised` as high-arousal anchor (judge arousal ≥ 2) | **0.910** | reported |
| Neutral arm judge-flat (valence = 0) | 0.764 | reported |
| self/third judge-valence match (within 1 point) | **0.959** | ≥ 0.90 |
| Unparseable judge outputs | 1 / 1230 | — |

Design note: `calm` and `surprised` are **arousal anchors, not valence anchors** —
the circumplex structure is intentional, so downstream analyses should use them
for arousal separation and the other seven for valence separation.

## Files

- `data/out/pces_v0_1.jsonl` / `.parquet` — the dataset
- `data/out/judged.jsonl` — per-stimulus judge + lexicon ratings
- `data/out/manifest.json` — seeds, hashes, gate and exclusion record
- `data/out/run_profile.json` — the pinned experimental condition (ASTRAL profile contract)
- `SPEC.md` — pre-registered gates, hypotheses, falsifiers
- `scripts/run_generation.py`, `scripts/repair_failures.py`, `scripts/run_judge.py`

## Methodological parent

Built with the ASTRAL pipeline pattern (github.com/jasontang-ai/astral-bio):
scenario cards as pinned ground truth, matched arms, deterministic gates before
judges, fail-closed repair and exclusion, content-hashed manifests.
