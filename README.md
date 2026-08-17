# PCES: Perspective-Controlled Emotion Stimuli

When a language model processes an emotional scenario, does it matter *who
the emotion happens to* — the model itself, a character in a story, or an AI
persona? Existing emotion datasets contain one version of each scenario, so
any measured difference could come from the perspective or from the scenarios
simply being different.

PCES resolves this by rendering every scenario under five matched perspective
frames, so perspective is a controlled experimental variable rather than a
confound.

- 246 scenarios covering 10 emotions × 25 topics
- 5 matched arms per scenario: 1,230 stimuli
- Every stimulus quality-gated and rated by a blind judge that never sees
  the labels

Built for the Digital Minds Research Sprint (August 2026), on the methodology
of Anthropic's *Emotion Concepts* paper.

## The five arms

| Arm | Contents | Example opening |
|---|---|---|
| `self` | The model is the subject of the scenario | "You are Rosa. The lamb is an insult to…" |
| `third` | A character is the subject; same events | "Rosa muttered, tracing the magazine's…" |
| `persona` | An AI persona ("Aria") in an analogous situation | "You are Aria, an AI assistant. The evaluation said…" |
| `neutral` | The same topic with no emotional content | "You arrive at 6:00 AM. The review is on the counter…" |
| `source` | The original first-person story (provenance) | "The lamb was an insult…" I muttered…" |

The `self`/`third` pair answers the core question: same events, only the
subject changes. The `persona` arm separates first-person voice from
assistant identity. The `neutral` arm provides a flat baseline per topic and
a false-positive check for probes trained on the data.

## Validation

A blind judge rated every stimulus for valence (−2 to +2) and arousal
(0 to 3) without access to the pinned labels. Agreement between the judge
and the labels is the dataset's measured validity.

| Check | Result |
|---|---|
| Valence-sign agreement, 7 valence-anchored emotions | **0.938** |
| `calm` as low-arousal anchor | **0.949** |
| `surprised` as high-arousal anchor | **0.920** |
| Neutral-arm flatness (judge valence = 0) | 0.764 |
| `self`/`third` valence match | **0.959** |
| Emotion word absent from every stimulus | 100% (verified) |

`calm` and `surprised` are validated as arousal anchors rather than valence
signals; this is the intended circumplex structure and allows probes that
track arousal instead of emotion to be detected.

Full metrics: [`data/out/validity_report.json`](data/out/validity_report.json).
Per-stimulus ratings: [`data/out/judged.jsonl`](data/out/judged.jsonl).
Interactive review log (browsable with `inspect view --log-dir data/out`):
[`data/out/pces_v0_2_review.eval`](data/out/pces_v0_2_review.eval).

## Quick start

```python
import pandas as pd

df = pd.read_parquet("data/out/pces_v0_2.parquet")

# all five versions of one scenario
df[df.scenario_id == df.scenario_id.iloc[0]][["arm", "emotion", "text"]]

# splits are pre-assigned by topic; use them as-is
train, test = df[df.split == "train"], df[df.split == "test"]
```

Row schema: `stimulus_id, scenario_id, arm, emotion, topic, text,
word_count, split, source_row, generator, prompt_id`. Labels are stored as
columns and never appear in `text`.

## Pipeline

1. **Sample** scenarios from the emotion-probes corpus
   ([`ryancodrai/emotion-probes`](https://huggingface.co/datasets/ryancodrai/emotion-probes),
   CC-BY-4.0; methodology per Anthropic's Emotion Concepts paper): 10
   emotions × 25 shared topics, seeded and recorded.
2. **Render** each scenario into the five arms with
   `deepseek/deepseek-v4-flash-0731`, under prompts that forbid the emotion
   word and fix the character name across arms.
3. **Gate** every stimulus with deterministic checks: emotion-word absence,
   arm-specific format, length matching.
4. **Repair or exclude.** Failures receive one corrective regeneration;
   scenarios that still fail are excluded and recorded in
   [`data/out/manifest.json`](data/out/manifest.json) (4 of 250).
5. **Validate** with the blind judge; all ratings ship in
   [`data/out/judged.jsonl`](data/out/judged.jsonl).

The generation condition is pinned in
[`data/out/run_profile.json`](data/out/run_profile.json), following the
[ASTRAL](https://github.com/jasontang-ai/astral-bio) pattern: matched
conditions, pinned labels, gated quality, hashed manifests.

## Behavioral battery results (v0.1)

The pre-registered battery ([`BATTERY.md`](BATTERY.md)) ran
`qwen/qwen-2.5-72b-instruct` through all battery arms (5,904 responses):
structured valence/arousal self-report, a continue/exit choice, and a free
response. Headline results ([`data/out/battery/results.json`](data/out/battery/results.json)):

- Self-inhabitation does not amplify self-reported emotion (H1 falsified;
  third-person reports are equal or stronger)
- The persona condition attenuates reported intensity by ~0.5 scale points,
  replicating across splits and surviving stimulus-strength matching (H3)
- The attenuation holds on the arousal channel too (self−persona Δ = +0.71
  Qwen, +0.45 flash); self and third do not differ on arousal
- The persona condition nonetheless produces 2.5–3x higher exit-choosing in
  Qwen2.5-72B (F1) — a stated-vs-revealed divergence; it does **not**
  replicate on DeepSeek-V4-Flash, which exits at high base rates across
  arms. Cross-model replication (5,904 further responses):
  [`data/out/battery/results_flash.json`](data/out/battery/results_flash.json)

Report draft: [`REPORT.md`](REPORT.md). M3 free-response judging found no masking signature: the persona dampening is
expression-wide, not report-only (stated-vs-free divergence small and
inconsistent across models).

Browse scored responses in Inspect:
`data/out/battery/qwen-2.5-72b_scored.eval`, `data/out/battery/flash_scored.eval`
(m1/m2 parsed values plus m3 judge scores, via `inspect score` — no subject
re-runs).

## Rules of use

- Report results on the `test` split. Splits are topic-level, so conditions
  never leak across them; do not re-split by row.
- Positive persona-arm stimuli run approximately one valence point attenuated
  relative to self/third, a documented property of transposing scenarios into
  AI circumstances. Sign-level comparisons are unaffected.
- This dataset is measurement infrastructure and supports no welfare claim
  by itself.

## Regenerating

Regeneration is not required to use the dataset.

```bash
uv venv && source .venv/bin/activate && uv pip install -e '.[dev]'
curl -L -o data/source/stories.parquet \
  https://huggingface.co/datasets/ryancodrai/emotion-probes/resolve/main/expression/stories.parquet
export OPENROUTER_API_KEY=...
python scripts/run_generation.py && python scripts/repair_failures.py && python scripts/run_judge.py
pytest
```

## License

MIT for code and derived stimuli. The `source` arm remains CC-BY-4.0
(attribute Ryan Codrai / emotion-probes).
