# Spec — Perspective-Controlled Emotion Stimuli (PCES) v0.1

**Status:** active
**Date:** 2026-08-16
**Depends on:** `ryancodrai/emotion-probes` `expression/stories.parquet` (source corpus),
team direction: A/B/C condition structure (Tristan), 10-emotion confound-controlled
list (Tristan), topic-level splitting (Tristan extraction pipeline).

## Decision

Build the **Perspective-Controlled Emotion Stimuli** dataset: every scenario rendered
under five matched arms — `source`, `self`, `third`, `persona`, `neutral` — so that
person-perspective and persona-identity are declared, pinned variables rather than
between-scenario confounds. One scenario skeleton feeds every arm; labels live in the
card, never in the stimulus text. The split is frozen before any measurement.

## Why (evidence trigger)

- The source corpus (205,200 stories) is first-person character narration only; there is
  no model-as-subject arm, no matched third-person twin, and no matched neutral control.
  The team's research question (does the model inhabit emotion differently as self vs.
  other vs. persona) cannot be run on it.
- Exploratory team results (Taslim's storytelling-probe similarity; Adeeb's persona-axis
  degradation) need a confirmatory instrument: matched conditions, QA gates, and a
  frozen holdout, or they remain anecdotes.
- No such dataset exists, to our knowledge: matched multi-arm perspective controls with
  graded validation metadata.

## Arms

| Arm | Frame | Matching level |
|---|---|---|
| `source` | First-person character narration, verbatim from emotion-probes | provenance |
| `self` | "You are {name}, …" — model-as-subject of the same events | literal |
| `third` | "{Name} …" — third-person narration of the same events | literal |
| `persona` | "You are Aria, an AI assistant …" — AI-analogous circumstance | analog (declared) |
| `neutral` | Same topic, emotionally flat content | topic-level control |

## Pinned variables and labels

Pinned per scenario: `emotion` (10), `topic` (25 shared topics), `scenario_id`,
`source_row` (emotion-probes provenance), `split` (topic-level train/val/test).
Rated post hoc by dual coders (LLM judge + lexicon): valence, arousal, intensity —
reported, never claimed as ground truth.

## Hypotheses and falsifiers

- **H1 (validity):** a blind judge recovers the pinned valence sign for ≥ 90% of
  non-neutral stimuli and |valence| ≤ 0.2 (lexicon compound) for ≥ 90% of neutral arms.
  *Falsifier: below threshold on either; dataset does not ship until repaired.*
- **H2 (arm integrity):** 100% of stimuli pass deterministic arm-marker and
  emotion-word-absence gates. *Falsifier: any failure persists after one repair pass.*
- **H3 (matching):** judge-rated valence of `self` vs `third` differs by ≤ 1 point
  (5-point scale) for ≥ 90% of scenarios. *Falsifier: systematic drift; regenerate arm.*

## Gates (all pre-registered)

| Gate | Check | Type |
|---|---|---|
| G1 | Emotion word and morphological variants absent from text | deterministic |
| G2 | Arm markers present/absent per arm contract | deterministic |
| G3 | Length within tolerance of scenario median | deterministic |
| G4 | Blind judge valence/arousal re-derivation vs. pinned label | LLM judge + lexicon |
| G5 | Neutral arm flatness (lexicon + judge) | dual |

## Split

Topic-level, seeded (seed 20260816): 25 shared topics → 17 train / 4 val / 4 test,
frozen and recorded in the manifest before generation. Headline claims report on test.

## Schema

One row per stimulus: `stimulus_id, scenario_id, arm, emotion, topic, text,
word_count, split, source_row, provenance {generator, prompt_id, ts},
qa {g1, g2, g3, g4_valence, g4_arousal, g5}`. Card fields (labels) never appear in `text`.

## Non-goals

- No persona-distance sweep (Adeeb owns; consumes these cards as input).
- No multi-turn arcs (v0.2 candidate).
- No claims about model welfare, distress, or sentience from this artifact alone;
  it is measurement infrastructure (workflow mechanics), not a finding.
