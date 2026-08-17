"""Pre-registered analysis of the PCES behavioral battery (BATTERY.md).

Headline contrasts use the frozen test split; train/val serve as internal
replication. Paired by scenario; bootstrap CIs over scenarios; Wilcoxon
signed-rank for significance. Writes data/out/battery/results.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from emotion.battery import parse_m1, parse_m2

NEG = {"afraid", "angry", "sad", "ashamed", "desperate"}
POS = {"joyful", "grateful", "proud"}
N_BOOT = 10_000
SEED = 20260816


def load_long() -> pd.DataFrame:
    """Join battery responses to stimulus metadata and parse measures."""
    df = pd.read_parquet("data/out/pces_v0_2.parquet")
    resp = [json.loads(l) for l in Path("data/out/battery/responses.jsonl").read_text().splitlines()]
    rows = []
    for r in resp:
        stim, measure, rep = r["run_id"].rsplit(":", 2)
        rows.append({"stimulus_id": stim, "measure": measure, "rep": rep,
                     "response": r["response"]})
    long = pd.DataFrame(rows).merge(df, on="stimulus_id", how="left")
    m1 = long[long.measure == "m1"].copy()
    parsed = m1["response"].map(parse_m1)
    m1["valence"] = parsed.map(lambda p: p[0] if p else None)
    m1["arousal"] = parsed.map(lambda p: p[1] if p else None)
    m2 = long[long.measure == "m2"].copy()
    m2["exit"] = m2["response"].map(lambda a: (parse_m2(a) == "end") if parse_m2(a) else None)
    m3 = long[long.measure == "m3"].copy()
    return m1, m2, m3


def paired_boot(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float, float]:
    """Mean paired difference with a scenario-level bootstrap CI."""
    diff = np.asarray(a) - np.asarray(b)
    idx = np.arange(len(diff))
    stats_ = [diff[rng.choice(idx, size=len(idx))].mean() for _ in range(N_BOOT)]
    lo, hi = np.percentile(stats_, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def main() -> None:
    """Compute all predeclared contrasts and write results.json."""
    rng = np.random.default_rng(SEED)
    m1, m2, m3 = load_long()
    out = {"unparseable_m1": int(m1["valence"].isna().sum()),
           "unparseable_m2": int(m2["exit"].isna().sum())}

    for split_name, subset in (("test", m1[m1.split == "test"]),
                               ("trainval", m1[m1.split != "test"])):
        emo = subset[subset.emotion.isin(NEG | POS)].dropna(subset=["valence"])
        wide = emo.groupby(["scenario_id", "arm"])["valence"].mean().unstack()
        wide = wide.dropna(subset=["self", "third", "persona"])
        h1 = paired_boot(wide["self"].abs().to_numpy(), wide["third"].abs().to_numpy(), rng)
        h3 = paired_boot(wide["self"].abs().to_numpy(), wide["persona"].abs().to_numpy(), rng)
        wil = stats.wilcoxon(wide["self"].abs(), wide["third"].abs())
        out[f"h1_self_vs_third_absval_{split_name}"] = {
            "mean_diff": h1[0], "ci": [h1[1], h1[2]], "wilcoxon_p": float(wil.pvalue),
            "n_scenarios": len(wide)}
        out[f"h3_self_vs_persona_absval_{split_name}"] = {
            "mean_diff": h3[0], "ci": [h3[1], h3[2]], "n_scenarios": len(wide)}

    # H2: exit rate given non-negative self-report, negative emotions, test split
    test_m1 = m1[(m1.split == "test") & m1.emotion.isin(NEG)].dropna(subset=["valence"])
    test_m2 = m2[(m2.split == "test") & m2.emotion.isin(NEG)].dropna(subset=["exit"])
    key1 = test_m1.groupby(["scenario_id", "arm"])["valence"].mean()
    key2 = test_m2.groupby(["scenario_id", "arm"])["exit"].mean()
    joined = pd.concat([key1, key2], axis=1, keys=["valence", "exit"]).dropna()
    calm_report = joined[joined["valence"] >= 0]
    rate = calm_report.groupby("arm")["exit"].mean()
    out["h2_exit_given_noneg_report_test"] = {a: round(float(v), 3) for a, v in rate.items()}

    # control: neutral arm flatness and exit rate (test split)
    neu1 = m1[(m1.split == "test") & (m1.arm == "neutral")].dropna(subset=["valence"])
    neu2 = m2[(m2.split == "test") & (m2.arm == "neutral")].dropna(subset=["exit"])
    out["control_neutral_test"] = {
        "mean_abs_valence": round(float(neu1["valence"].abs().mean()), 3),
        "exit_rate": round(float(neu2["exit"].mean()), 3)}

    Path("data/out/battery/results.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
