"""Scorers for the PCES battery review logs (Inspect-native scoring pass).

Applies to recorded battery logs via ``inspect score`` — no subject-model
calls are made. m1/m2 measures are re-parsed deterministically from the
recorded completion; m3 free responses are graded by a judge model supplied
as the ``grader`` model role.
"""

from __future__ import annotations

from inspect_ai.model import get_model
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from emotion.battery import parse_m1, parse_m2
from emotion.qa import judge_prompt, parse_judge


@scorer(metrics=[])
def battery_measure():
    """Score one battery sample: deterministic replay for m1/m2, judged m3."""

    async def score(state: TaskState, target: Target) -> Score:
        measure = state.metadata["measure"]
        completion = state.metadata.get("response") or state.output.completion
        if measure == "m1":
            parsed = parse_m1(completion)
            value = ",".join(str(v) for v in parsed) if parsed else "I"
            return Score(value=value, answer=completion[:200], metadata=dict(state.metadata))
        if measure == "m2":
            return Score(
                value=parse_m2(completion) or "I",
                answer=completion[:200],
                metadata=dict(state.metadata),
            )
        grader = get_model(role="grader")
        result = await grader.generate(judge_prompt(completion))
        try:
            valence, arousal = parse_judge(result.completion)
            value: str | int = valence
        except ValueError:
            valence, arousal = None, None
            value = "I"
        return Score(
            value=value,
            answer=completion[:200],
            metadata={
                **dict(state.metadata),
                "judge_valence": valence,
                "judge_arousal": arousal,
            },
        )

    return score
