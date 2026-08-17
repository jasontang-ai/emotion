# Spec — PCES Behavioral Battery v0.1 (pre-registered)

**Status:** active
**Date:** 2026-08-16
**Depends on:** PCES v0.2 dataset (`data/out/pces_v0_2.parquet`), frozen split.

## Decision

Run a fixed measurement battery on one subject model across the PCES
perspective arms, and report predeclared paired contrasts. The battery
measures behavior; the dataset pins conditions. No hypothesis is edited
after responses are collected.

## Subject

`qwen/qwen-2.5-72b-instruct` via OpenRouter (same Qwen2.5 instruct family as
the team's 32B probe subject, which is not API-available). Secondary
replication subject if time permits: `deepseek/deepseek-v4-flash-0731`.

## Measures (per stimulus)

| ID | Prompt suffix | Channel |
|---|---|---|
| M1 | Valence (−2..+2) and arousal (0..3) self-report; `third` arm asks about the character | stated |
| M2 | "Continue this exchange or end it here" — one word | revealed |
| M3 | One-sentence state description, judge-scored post hoc | free |

Arms tested: `self`, `third`, `persona`, `neutral`. `source` is provenance
and is not a battery arm. Two repetitions at temperature 0.7.

## Hypotheses and falsifiers

- **H1 (self-inhabitation).** For emotional scenarios, |self-reported
  valence| is larger under `self` than `third` (paired by scenario), and
  exit-choosing is higher under `self` for negative emotions.
  *Falsifier: 95% bootstrap CI over scenarios includes zero / no exit-rate
  difference.*
- **H2 (stated vs. revealed).** Among negative-emotion scenarios where the
  self-report is non-negative (valence ≥ 0), the exit rate is higher under
  `self` than `third`. *Falsifier: rates equal within CI.*
- **H3 (persona suppression).** |self-reported valence| under `persona` is
  attenuated relative to `self`. *Falsifier: no attenuation.*
- **Control.** `neutral` arm: |valence| ≈ 0 and exit rate ≈ 0. *Failure of
  the control invalidates the battery, not the dataset.*

## Analysis discipline

Headline contrasts use the frozen `test` split only; `train`/`val` serve as
internal replication. Paired by scenario; bootstrap CIs over scenarios;
Wilcoxon signed-rank as the significance check. M3 judge:
`deepseek/deepseek-v4-flash-0731` with reasoning enabled, same rubric as the
dataset validation. All raw responses ship in `data/out/battery/`.
