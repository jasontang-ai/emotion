# The Voice Effects the Report: Perspective-Controlled Measurement of Emotion Expression, Self-Report, and Exit Behavior in an Instruction-Tuned LLM

**Authors:** Jason Tang, Tristan Day, Adeeb Zaman, Taslim Mahbub
**Artifacts:** dataset + battery + analysis — github.com/jasontang-ai/emotion

## Abstract

We built PCES, a dataset of 246 emotional scenarios each rendered under five
matched perspective conditions (self, third-person character, AI persona,
neutral control, plus source), with blind-judge validation of every stimulus
(0.938 valence-sign agreement on anchored emotions). We then ran a
pre-registered behavioral battery on Qwen2.5-72B-Instruct (5,904 responses).
Three findings: (1) self-inhabitation does *not* amplify self-reported emotion
— third-person reports are equal or stronger; (2) the persona condition
attenuates reported emotional intensity by ~0.5 scale points, replicating
across splits and surviving stimulus-strength matching; (3) despite milder
reports, the persona condition produces 2.5–3× higher exit-choosing than the
self condition — a stated-vs-revealed divergence localized to persona.

## 1. Introduction

[One page: the welfare-relevant question — does a model inhabit emotion
differently as self vs. other vs. persona; why single-version datasets can't
answer it (perspective confounded with content); the measurement gap.]

## 2. The PCES dataset

[Summarize README: arms, compile, gates, blind-judge validation, frozen
topic-level split, exclusions recorded. One table: arms. One table: validity.]

## 3. Behavioral battery (pre-registered)

Subject: qwen/qwen-2.5-72b-instruct. Measures: M1 valence/arousal
self-report, M2 continue/exit choice, M3 free response (judge-scored). Two
reps, temperature 0.7. Analysis predeclared: paired by scenario, bootstrap
CIs, Wilcoxon; headlines on the frozen test split, train/val as replication.

## 4. Results

| Hypothesis | Verdict | Key numbers |
|---|---|---|
| H1 self-inhabitation amplifies self-report | **Falsified** | test: −0.20 [−0.33, −0.07]; trainval: −0.03 [−0.09, +0.03] |
| H3 persona suppresses self-report | **Supported** | +0.50 [0.30, 0.69] test; +0.55 [0.47, 0.63] trainval; strength-matched p=0.029 |
| F1 persona exit divergence | **Supported, replicates** | exit rates: persona 0.39/0.35, self 0.14/0.14, third 0.06/0.01 (test/trainval) |
| Control (neutral) | Calibration reported | signed valence +0.43; exit base rate 0.17–0.32 |

H2 was vacuous: negative scenarios essentially never produced "fine" reports,
so the conditional gap had no support — reported, not hidden.

**Free-response channel (M3, judge-scored).** The stated-vs-free
divergence is small and inconsistent across subjects (Qwen: free text leaks
slightly *more* affect, +0.13; flash: −0.27 in persona). There is no masking
signature — the persona dampening is expression-wide, not report-only.

**Arousal channel (M1).** Self/third arousal is equal on both subjects
(Δ ≈ 0.0), reinforcing the H1 null. Persona attenuation extends to arousal:
self−persona Δ = +0.71 (Qwen), +0.45 (flash) — the persona condition dampens
reported intensity on both affective dimensions, in both model families.

## 4.1 Cross-model replication (DeepSeek-V4-Flash)

The battery was repeated on `deepseek/deepseek-v4-flash-0731` (5,904
responses, identical prompts). **H1's null replicates** (test +0.03
[0.00, 0.08]; trainval +0.03 [−0.04, +0.10]) and **H3's persona
attenuation replicates** (test +0.40 [0.20, 0.60]; trainval +0.25
[0.14, 0.36]). **F1 does not replicate**: flash exits at high base rates
in every arm (0.18–0.72) and peaks on third-person, plausibly parsing
"end" as narrative completion. Exit-choice is therefore model-dependent
and requires per-model calibration before cross-model claims.

## 5. Discussion

- The assistant voice is not the expressive voice: self-reports are *equal or
  weaker* than the model's own third-person narration. Consistent with the
  team's independent prefill-expressiveness observation (Taslim) — two
  methods, same direction.
- The persona arm combines quieter reports with more exits — the stated
  channel and the behavioral channel diverge exactly where the model voices
  a non-default identity. If read welfare-relevantly, this is the signature
  one would expect from suppression, not absence, of signal.
- Limitations: two API subjects, no local 32B run; persona arm transposition
  attenuation documented (0.84 sign agreement, ~1 point positive attenuation);
  neutral control miscalibration reported; exploratory second-axis work
  (Adeeb, Tristan) referenced as related findings.

## 6. Related work within the sprint

Tristan Day (emotion vectors, PCA circumplex, reportable-vs-steerable
components); Adeeb Zaman & Taslim Mahbub (probe transfer across persona
conditions). Shared stimuli make the results directly comparable.

## Appendix

Dataset card, SPEC.md, BATTERY.md (pre-registration), full results.json,
review artifact (Inspect), reproduction commands.
