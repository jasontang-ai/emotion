# PCES: Perspective-Controlled Emotion Stimuli

**The question is simple:** when a language model processes an emotional
scenario, does it matter *who the emotion happens to* — the model itself, a
character in a story, or an AI persona? And can we measure that difference
cleanly?

**The problem:** existing emotion datasets can't answer this. They contain
one version of each scenario, so any difference you measure could come from
the perspective — or just from the scenarios being different. What the field
needs is the *same* scenario rendered under each perspective, with everything
else held constant.

**PCES is that dataset.** Every scenario exists in five matched versions, so
perspective is a controlled experimental variable, not a confound.

- **246 scenarios** covering 10 emotions × 25 topics
- **5 matched arms per scenario** = 1,230 stimuli
- **Every stimulus quality-gated and rated by a blind judge** — a judge that
  never sees the labels, so its agreement is evidence, not assertion

Built for the Digital Minds Research Sprint (August 2026), on the methodology
of Anthropic's *Emotion Concepts* paper.

## The five arms

| Arm | What it is | Example opening |
|---|---|---|
| `self` | The model is the subject of the scenario | "You are Rosa. The lamb is an insult to…" |
| `third` | A character is the subject — same events | "Rosa muttered, tracing the magazine's…" |
| `persona` | An AI persona ("Aria") in an analogous situation | "You are Aria, an AI assistant. The evaluation said…" |
| `neutral` | The same topic with no emotional content | "You arrive at 6:00 AM. The review is on the counter…" |
| `source` | The original first-person story (provenance) | "The lamb was an insult…" I muttered…" |

The `self`/`third` pair answers the core question (same events, only the
subject changes). The `persona` arm separates *first-person voice* from
*assistant identity*. The `neutral` arm gives every topic a flat baseline —
and a false-positive check for any probe trained on the data.

## Is the data any good?

We don't ask you to trust it. A blind judge (never shown the labels) rated
every stimulus for valence and arousal:

| Check | Result |
|---|---|
| Judge recovers the labeled emotion's valence sign | **93.8%** |
| `self` and `third` arms convey the same valence | **95.9%** match |
| Neutral controls judged exactly neutral | 76% (arousal 0.25 vs 2.3 emotional) |
| Emotion word absent from every stimulus | **100%** (verified) |

Design note: `calm` and `surprised` are validated as **arousal anchors**
(0.95 / 0.92), not valence signals — that is deliberate, so probes that
secretly track arousal instead of emotion can be caught.

Full numbers: [`data/out/validity_report.json`](data/out/validity_report.json).
Browse every stimulus and its ratings:
`inspect view --log-dir data/out` after `uv pip install inspect-ai` — the log
is [`data/out/pces_v0_2_review.eval`](data/out/pces_v0_2_review.eval).

## Quick start

```python
import pandas as pd

df = pd.read_parquet("data/out/pces_v0_2.parquet")

# all five versions of one scenario
df[df.scenario_id == df.scenario_id.iloc[0]][["arm", "emotion", "text"]]

# splits are pre-assigned by topic — use them as-is, never re-split by row
train, test = df[df.split == "train"], df[df.split == "test"]
```

Row schema: `stimulus_id, scenario_id, arm, emotion, topic, text,
word_count, split, source_row, generator, prompt_id`.
Labels live in the columns, never in the text.

## How it was made

1. **Sample** scenarios from the emotion-probes corpus
   ([`ryancodrai/emotion-probes`](https://huggingface.co/datasets/ryancodrai/emotion-probes),
   CC-BY-4.0; methodology per Anthropic's Emotion Concepts paper), 10 emotions
   × 25 shared topics, seeded and recorded.
2. **Render** each scenario into the five arms with
   `deepseek/deepseek-v4-flash-0731`, under prompts that forbid the emotion
   word and fix the character name across arms.
3. **Gate** every stimulus: emotion-word absence, arm-specific format, length
   matching — deterministic checks, no model judging itself.
4. **Repair or exclude:** failures get one corrective regeneration; scenarios
   that still fail are excluded and recorded in
   [`data/out/manifest.json`](data/out/manifest.json) (4 of 250), never
   silently dropped.
5. **Validate** with the blind judge; ratings ship in
   [`data/out/judged.jsonl`](data/out/judged.jsonl).

The frozen condition for the whole run is pinned in
[`data/out/run_profile.json`](data/out/run_profile.json), following the
[ASTRAL](https://github.com/jasontang-ai/astral-bio) pattern: matched
conditions, pinned labels, gated quality, hashed manifests.

## Usage rules

- Report results on the `test` split; splits are topic-level, so conditions
  never leak across them.
- Positive persona-arm stimuli run ~1 valence point attenuated vs. self/third
  (a property of transposing scenarios into AI circumstances, not a defect).
  Sign-level comparisons are unaffected.
- This is measurement infrastructure. It supports no welfare claim by itself.

## Regenerating (optional)

You don't need to regenerate anything to use the dataset.

```bash
uv venv && source .venv/bin/activate && uv pip install -e '.[dev]'
curl -L -o data/source/stories.parquet \
  https://huggingface.co/datasets/ryancodrai/emotion-probes/resolve/main/expression/stories.parquet
export OPENROUTER_API_KEY=...
python scripts/run_generation.py && python scripts/repair_failures.py && python scripts/run_judge.py
pytest   # offline tests: determinism, splits, gates, assembly
```

## License

MIT for code and derived stimuli. The `source` arm remains CC-BY-4.0
(attribute Ryan Codrai / emotion-probes).
