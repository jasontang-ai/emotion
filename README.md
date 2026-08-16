# PCES — Perspective-Controlled Emotion Stimuli

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**PCES** is a matched-conditions emotion dataset: every scenario is rendered
under five controlled perspective frames, so *who the emotion happens to* is a
pinned experimental variable rather than a confound.

Built for the Digital Minds Research Sprint (2026-08) to answer: does a model
inhabit an emotion differently as **itself**, as a **character**, or as a
**persona**?

## The dataset

```text
one scenario card (emotion × topic, pinned)
→ five matched stimuli
```

| Arm | Frame | Matching |
|---|---|---|
| `source` | First-person character narration, verbatim from emotion-probes | provenance |
| `self` | "You are {Name}, …" — the model as subject, same events | literal |
| `third` | "{Name} …" — third-person narration, same events | literal |
| `persona` | "You are Aria, an AI assistant …" — AI-analogous circumstance | analog (declared) |
| `neutral` | Same topic, emotionally flat | topic control |

- **246 scenarios × 5 arms = 1,230 stimuli** · 10 emotions × 25 topics
- **Split frozen before measurement:** topic-level 17/4/4 train/val/test
- **Labels are metadata, never text:** the emotion word is gate-verified absent
  from every stimulus

### Files

| File | Contents |
|---|---|
| [`data/out/pces_v0_2.parquet`](data/out/pces_v0_2.parquet) | The dataset (488 KB) |
| [`data/out/pces_v0_2.jsonl`](data/out/pces_v0_2.jsonl) | Same rows, JSONL |
| [`data/out/judged.jsonl`](data/out/judged.jsonl) | Per-stimulus blind-judge + lexicon ratings |
| [`data/out/manifest.json`](data/out/manifest.json) | Seeds, hashes, QA and exclusion record |
| [`data/out/validity_report.json`](data/out/validity_report.json) | Aggregate validity metrics |
| [`data/out/run_profile.json`](data/out/run_profile.json) | The pinned generation condition (ASTRAL profile contract) |
| [`data/out/pces_v0_2_review.eval`](data/out/pces_v0_2_review.eval) | Inspect-native review log of the judge validation run (1,230 samples) |

Row schema: `stimulus_id, scenario_id, arm, emotion, topic, text, word_count,
split, source_row, generator, prompt_id`.

## Quick start

```python
import pandas as pd

df = pd.read_parquet("data/out/pces_v0_2.parquet")

# One scenario, all five arms:
df[df["scenario_id"] == df["scenario_id"].iloc[0]][["arm", "emotion", "text"]]

# Train/test is already assigned — never re-split by row:
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
```

## Validity

A blind judge (never shown labels) re-rates every stimulus. Full breakdown in
[`data/out/validity_report.json`](data/out/validity_report.json):

| Check | Result |
|---|---|
| Valence-sign agreement, 7 valence-anchored emotions | **0.938** |
| `calm` as low-arousal anchor | **0.949** |
| `surprised` as high-arousal anchor | **0.920** |
| Neutral-arm flatness | 0.764 |
| self/third valence match | **0.959** |
| persona sign agreement (v0.2) | 0.842 |

Design note: `calm` and `surprised` are **arousal anchors, not valence
anchors** — that is the circumplex structure working as intended.

**v0.2 change.** An audit found the v0.1 persona frame ("no legal rights, may
be shut down…") dragged positive-valence persona stimuli to zero or below — a
frame×valence confound. The persona arm was regenerated with a valence-neutral
frame ("You are Aria, an AI assistant"); only changed rows were re-judged
(flagged `rejudged_v0_2` in `judged.jsonl`). Positive persona stimuli now hold
correct sign but run ~1 point attenuated vs. self/third — a documented arm
property of the AI-circumstance transposition, not a defect. The constrained
life-circumstances persona belongs in a variant arm, not the baseline.

## Regenerating (optional)

You do not need to regenerate anything to use the dataset. To rebuild:

```bash
uv venv && source .venv/bin/activate
uv pip install -e '.[dev]'

# Source corpus (CC-BY-4.0) into data/source/:
curl -L -o data/source/stories.parquet \
  https://huggingface.co/datasets/ryancodrai/emotion-probes/resolve/main/expression/stories.parquet

export OPENROUTER_API_KEY=...
python scripts/run_generation.py      # ~1,000 calls, bounded concurrency, checkpoints
python scripts/repair_failures.py     # one corrective pass over gate failures
python scripts/run_judge.py           # blind-judge validity report
```

## Review the dataset in the Inspect viewer

The `.eval` file is a **review artifact**: it replays the recorded judge
ratings against the pinned labels so you can browse every stimulus, arm, and
rating interactively. It carries no product score of its own (same convention
as ASTRAL's Gold100 viewer log).

```bash
uv pip install inspect-ai
inspect view --log-dir data/out
```

## Verify

```bash
pytest        # 8 offline tests: determinism, splits, gates, assembly
```

## Rules of use

- Report headline claims on the `test` split only.
- Do not re-split by row; splits are topic-level on purpose (a topic appears in
  exactly one split, so conditions never leak across it).
- Four scenarios failed QA twice and were **excluded** — they are recorded in
  [`data/out/manifest.json`](data/out/manifest.json), not silently dropped.
- This is measurement infrastructure. It supports no welfare claim by itself.

## Provenance and license

- Source corpus: [`ryancodrai/emotion-probes`](https://huggingface.co/datasets/ryancodrai/emotion-probes)
  (CC-BY-4.0), methodology per Sofroniew et al., *Emotion Concepts and their
  Function in a Large Language Model* (Anthropic, 2026).
- Generation: `deepseek/deepseek-v4-flash-0731` via OpenRouter.
- Method pattern: [ASTRAL](https://github.com/jasontang-ai/astral-bio) —
  pinned cards, matched arms, fail-closed gates, hashed manifests.
- Pre-registered design: [`SPEC.md`](SPEC.md).
