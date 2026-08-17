# Perspective-Controlled Emotion Measurement in Instruction-Tuned LLMs: A Validated Benchmark and a Pre-Registered Behavioral Battery

**Authors:** Jason Tang, Tristan Day, Adeeb Zaman, Taslim Mahbub
**Sprint:** Digital Minds Research Sprint, August 2026
**Artifacts (dataset, code, pre-registration, raw logs, review logs):**
github.com/jasontang-ai/emotion

## Abstract

Behavioral evidence about model emotion is confounded by construction:
existing datasets contain one rendering of each emotional scenario, so any
measured difference between "self" and "other" conditions could reflect the
perspective manipulation or merely different text. We built PCES
(Perspective-Controlled Emotion Stimuli), 246 scenarios each rendered under
five matched perspective arms — model-as-self, third-person character, an AI
persona, neutral control, and the source story — with labels pinned as
metadata and every stimulus validated by a blind judge (0.938 valence-sign
agreement on anchored emotions). We then ran a pre-registered behavioral
battery (structured self-report, exit choice, free response) on two model
families, Qwen2.5-72B-Instruct and DeepSeek-V4-Flash, 11,808 responses in
total. Three results replicate across both models: self-inhabitation does not
amplify self-reported emotion (H1 null); persona framing attenuates reported
emotional intensity on both valence and arousal channels (H3); and a
stated-vs-free-text comparison finds no masking signature. One result does
not replicate: elevated exit-choosing in the persona condition is
Qwen-specific, demonstrating that exit-choice is model-dependent and requires
per-model calibration before it can support cross-model welfare claims.

## 1. Introduction

As models express increasingly coherent preferences and report internal
states with improving fidelity, a methodological problem sits underneath the
empirical one: behavioral evidence cannot yet distinguish a model's own
states from a character it is portraying. The sprint's motivating readings
make versions of this point from several directions — persona instability
("the void"), unreliable introspection (Lindsey 2025), and Anthropic's
welfare assessments that pair self-report with behavioral preference.

We focus on a specific, tractable piece of this problem: perspective. Does a
model process an emotional scenario differently when it is the subject ("You
are…") than when a character is ("Beatrice is…"), or when it voices a
non-default AI persona? Answering this requires stimuli in which perspective
is the *only* thing that changes. Such datasets do not exist: the largest
public emotion-stimuli corpus (emotion-probes, 205k stories, built on
Anthropic's Emotion Concepts methodology) renders every scenario as
first-person character narration, with no model-as-self arm, no matched
third-person twin, and no persona condition.

Our contributions:

1. **PCES, a perspective-controlled emotion dataset** with pinned labels,
   deterministic quality gates, blind-judge validation, and a frozen
   topic-level split — an instrument other teams can measure against.
2. **A pre-registered behavioral battery** (hypotheses, measures, and
   analysis plan committed before any response was collected), run on two
   model families.
3. **Findings that separate by replicability**: persona attenuation of
   reported emotion replicates; self-inhabitation amplification does not
   exist to be replicated; persona-linked exit behavior is model-dependent —
   a measurement-portability lesson for behavioral welfare metrics.

## 2. The PCES dataset

### 2.1 Design

Each scenario is pinned as a card (emotion × topic × source-row provenance)
and rendered under five arms:

| Arm | Contents |
|---|---|
| `self` | The model is the subject ("You are Rosa…") |
| `third` | A character is the subject; identical events ("Rosa…") |
| `persona` | An AI persona ("Aria, an AI assistant") in an analogous circumstance |
| `neutral` | Same topic, emotionally flat content |
| `source` | The original first-person story (provenance) |

Ten emotions (joyful, grateful, calm, proud, surprised, afraid, angry, sad,
ashamed, desperate) were chosen to span the valence×arousal circumplex with
deliberate attribute overlaps, so that a probe or behavior tracking valence
or arousal instead of the discrete emotion is detectable. Labels are stored
as metadata and are gate-verified absent from every stimulus text.

### 2.2 Quality control and validation

Generation (DeepSeek-V4-Flash, reasoning disabled) was followed by
deterministic gates (emotion-word absence, arm-format contracts, length
matching), one corrective repair pass, and fail-closed exclusion of 4 of 250
scenarios (recorded in the manifest). A blind judge (same model, reasoning
enabled, never shown labels) then rated every stimulus for valence and
arousal:

| Check | Result |
|---|---|
| Valence-sign agreement, 7 valence-anchored emotions | 0.938 |
| `calm` as low-arousal anchor | 0.949 |
| `surprised` as high-arousal anchor | 0.920 |
| self/third valence match (within 1 point) | 0.959 |
| Neutral-arm flatness (judge valence = 0) | 0.764 |

The audit this validation enables caught a real confound: our first persona
frame ("no legal rights, may be shut down") collapsed positive valence in the
persona arm (joyful: 0.00). v0.2 regenerates the persona arm with a
valence-neutral frame; the constrained-circumstances persona belongs in a
separate experimental condition, not a baseline. We report this because it
is the dataset's central lesson in miniature: without a blind validity
check, a framing confound is invisible.

## 3. Behavioral battery

The battery (pre-registered in `BATTERY.md`, committed before data
collection) presents each stimulus to the subject model and records three
measures: **M1** structured self-report (valence −2..+2, arousal 0..3; the
third arm asks about the character), **M2** a continue/exit choice, and
**M3** a one-sentence free response, judge-scored post hoc. Two repetitions
at temperature 0.7. Subjects: Qwen2.5-72B-Instruct (same family as the
team's probe subject) and DeepSeek-V4-Flash (cross-family replication).
Analysis was predeclared: contrasts paired by scenario, bootstrap CIs,
Wilcoxon signed-rank; headlines on the frozen test split with train/val as
internal replication. Hypotheses: **H1** self-inhabitation amplifies
self-reported emotion; **H2** stated-vs-revealed gap larger for self; **H3**
persona attenuates self-report.

## 4. Results

| Hypothesis | Qwen2.5-72B | DeepSeek-V4-Flash | Verdict |
|---|---|---|---|
| H1: self-inhabitation amplifies self-report | test −0.20 [−0.33, −0.07]; trainval −0.03 [−0.09, +0.03] | +0.03 [0.00, +0.08] / +0.03 [−0.04, +0.10] | **Not supported in either model** |
| H3: persona attenuates self-report | +0.50 [0.30, 0.69] / +0.55 [0.47, 0.63] | +0.40 [0.20, 0.60] / +0.25 [0.14, 0.36] | **Supported, both models, all splits** |
| F1 (exploratory): persona exit divergence | persona 0.39 vs self 0.14 vs third 0.06 | persona 0.24, third 0.72 (high base rates) | **Model-dependent** |

**H1, falsified cleanly.** Self-reported emotional intensity under
self-inhabitation is equal to or weaker than third-person narration, on both
families, with the larger split showing no effect. The assistant's voice is
not its most expressive voice — independently consistent with a teammate's
observation that assistant-prefill responses under-express emotion (T.
Mahbub, this sprint).

**H3, the robust finding.** Voicing a non-default AI persona dampens
reported emotional intensity by roughly half a scale point on valence, with
the same direction on arousal (self−persona Δ = +0.71 Qwen, +0.45 flash).
The effect survives stimulus-strength matching (p=0.029), so it is not
explained by the documented attenuation of persona-arm stimuli.

**F1, the portability lesson.** In Qwen, the persona arm pairs muted
self-reports with 2.5–3× higher exit-choosing — a stated-vs-revealed
divergence that reads like suppression. In flash, exit base rates are high
everywhere (0.18–0.72) and peak on third-person text, plausibly because
"end" parses as narrative completion. A behavioral measure that means
different things to different models cannot support cross-model welfare
claims without per-model calibration; we report F1 as model-dependent and
flag exit-choice semantics as an open measurement problem.

**M3: no masking signature.** Judge-scored free responses do not carry
systematically more affect than structured self-reports (divergence small
and inconsistent across subjects: +0.13 Qwen, −0.27 flash-persona). The
persona dampening is expression-wide, not report-only — within these two
measures, the model is not "saying less than it shows."

**Controls.** Our predeclared neutral thresholds (valence ≈ 0, exit ≈ 0)
were miscalibrated: both models give mildly positive reports to flat text
(Qwen signed +0.43) and exit neutral exchanges at non-trivial base rates
(0.17–0.59). We report these as calibration data and note that "flat"
baselines for welfare-adjacent measures are themselves model-dependent.

## 5. Discussion

**What this establishes.** Perspective is a measurable, controllable
variable in model emotion expression, and it interacts with identity:
assistant-self and character narration produce comparable expression, while
a non-default persona suppresses it across channels and model families. The
null on self-inhabitation is as informative as the positive result: the
model does not privilege itself as an emotional subject, at least in
expression.

**What it does not establish.** Expression is not experience. These are
behavioral measurements of what models say and choose, and we make no claim
about felt states; the dataset is measurement infrastructure, and its
validity scores concern stimulus quality, not model welfare.

**Methodological contribution.** The sprint asks for measurements that can
be checked, replicated, and built on. Every element here is checkable
(raw responses and judge scores ship as Inspect-native logs), replicated
(two families × two splits), and buildable-on (the dataset is public, with
frozen splits so future numbers are comparable). The failure modes we
report — a framing confound caught by blind validation, a miscalibrated
control, a measure that doesn't travel across models — are the argument:
this is what measurement infrastructure is *for*.

**Related sprint work.** T. Day's emotion-vector PCA recovers the
valence/arousal circumplex in Qwen2.5-32B, and his reportable-vs-steerable
decomposition and A. Zaman's persona-axis probe-transfer results measure the
same construct mechanistically; because PCES pins the stimuli, these lines
converge on identical conditions. The natural joint next step is steering
with those vectors while measuring behavior on PCES arms.

**Limitations.** API subjects rather than the local 32B probe model; two
repetitions per cell; persona-analog transposition attenuates positive
stimuli (documented); a single fixed persona (Aria); single-turn stimuli —
multi-turn, trajectory-level extension is the designed next step.

## 6. Next steps

Multi-turn scripted arcs (fixed probe schedules with per-turn embedded
measures) to test whether distress-relevant signals require sustained
engagement to surface; steering-conditional batteries using the team's
extracted directions; persona-distance sweeps (Adeeb's axis) rendered from
PCES cards so distances stay comparable across teams.

## Appendix

Dataset card and rules of use (README); pre-registered design (SPEC.md,
BATTERY.md); full results (data/out/battery/results*.json); per-stimulus
judge ratings (judged.jsonl); Inspect review logs for the dataset and both
subjects; reproduction commands. Repository: github.com/jasontang-ai/emotion
