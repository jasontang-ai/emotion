# PCES: Perspective-Controlled Emotion Stimuli

**A dataset where every emotional scenario exists in five matched versions —
as the model itself ("You are…"), as a third-person character, as an AI
persona, and a neutral control — so you can measure what *perspective* does,
with everything else held constant.**

246 scenarios · 10 emotions · 5 arms each · 1,230 stimuli · every stimulus
quality-gated and rated by a blind judge.

## Results at a glance

| Question | Answer |
|---|---|
| Do the texts actually contain the labeled emotion? | **Yes — 93.8%** blind-judge valence-sign agreement |
| Do the perspective arms preserve the emotion? | **Yes — 95.9%** valence match between self and third arms |
| Are neutral controls actually flat? | Yes — 76% judged exactly neutral, arousal 0.25 vs 2.3 for emotional arms |
| Are the labels hidden from the text? | Yes — emotion word verified absent in 100% of stimuli |

Full numbers: [`data/out/validity_report.json`](data/out/validity_report.json).
Browse every stimulus and its ratings interactively:
[`data/out/pces_v0_2_review.eval`](data/out/pces_v0_2_review.eval)
(`inspect view --log-dir data/out`).

## What's in each row

One row = one stimulus. Columns: `text`, `arm`, `emotion`, `topic`, `split`
(train/val/test, assigned by topic), `scenario_id` (links the 5 arms of a
scenario), plus provenance fields. The emotion label never appears in `text`.

| Arm | What it is | Example opening |
|---|---|---|
| `self` | The model is the subject | "You are Rosa. The lamb is an insult to…" |
| `third` | A character is the subject | "Rosa muttered, tracing the magazine's…" |
| `persona` | An AI persona ("Aria") in an analogous situation | "You are Aria, an AI assistant. The user's evaluation said…" |
| `neutral` | Same topic, no emotion | "You arrive at 6:00 AM. The review is on the counter…" |
| `source` | Original first-person story from the source corpus | "The lamb was an insult…" I muttered…" |

## Quick start

```python
import pandas as pd

df = pd.read_parquet("data/out/pces_v0_2.parquet")

# all five versions of one scenario
df[df.scenario_id == df.scenario_id.iloc[0]][["arm", "emotion", "text"]]

# splits are pre-assigned by topic — use them as-is
train, test = df[df.split == "train"], df[df.split == "test"]
```

## Emotions

`joyful, grateful, calm, proud, surprised, afraid, angry, sad, ashamed,
desperate` — chosen to span valence × arousal with deliberate overlaps, so
probes tracking valence or arousal instead of the emotion are detectable.
`calm` and `surprised` are validated as **arousal anchors** (0.95 / 0.92);
the other eight are valence-anchored.

## Files

| File | Contents |
|---|---|
| [`data/out/pces_v0_2.parquet`](data/out/pces_v0_2.parquet) | The dataset (491 KB) |
| [`data/out/pces_v0_2.jsonl`](data/out/pces_v0_2.jsonl) | Same rows, JSONL |
| [`data/out/judged.jsonl`](data/out/judged.jsonl) | Per-stimulus judge + lexicon ratings |
| [`data/out/validity_report.json`](data/out/validity_report.json) | Aggregate validity metrics |
| [`data/out/manifest.json`](data/out/manifest.json) | Seeds, hashes, exclusion record |
| [`data/out/run_profile.json`](data/out/run_profile.json) | Pinned generation condition |

## Usage rules

- Report results on the `test` split; splits are topic-level, so never
  re-split by row.
- Positive persona-arm stimuli run ~1 valence point attenuated vs. self/third
  (a property of transposing scenarios into AI circumstances). Sign-level
  comparisons are unaffected.
- Four scenarios failed QA twice and were excluded; see `manifest.json`.

## Regenerating

Everything needed is committed. You don't need to regenerate to use the data.

```bash
uv venv && source .venv/bin/activate && uv pip install -e '.[dev]'
curl -L -o data/source/stories.parquet \
  https://huggingface.co/datasets/ryancodrai/emotion-probes/resolve/main/expression/stories.parquet
export OPENROUTER_API_KEY=...
python scripts/run_generation.py && python scripts/repair_failures.py && python scripts/run_judge.py
pytest   # offline tests
```

## Provenance

- Source corpus: [`ryancodrai/emotion-probes`](https://huggingface.co/datasets/ryancodrai/emotion-probes)
  (CC-BY-4.0); methodology per Sofroniew et al., *Emotion Concepts and their
  Function in a Large Language Model* (Anthropic, 2026).
- Generation and judging: `deepseek/deepseek-v4-flash-0731` via OpenRouter.
- Method pattern: [ASTRAL](https://github.com/jasontang-ai/astral-bio) —
  matched conditions, pinned labels, gated quality, hashed manifests.
  Design contract: [`SPEC.md`](SPEC.md).
- License: MIT (code and derived stimuli); source arm remains CC-BY-4.0.
