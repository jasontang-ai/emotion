
"""Independent audit of the shipped PCES v0.1 dataset."""
import json
import re
from collections import Counter

import pandas as pd

df = pd.read_parquet("data/out/pces_v0_1.parquet")
report = {}

# --- 1. Schema integrity -------------------------------------------------
required = ["stimulus_id","scenario_id","arm","emotion","topic","text",
            "word_count","split","source_row","generator","prompt_id"]
report["missing_cols"] = [c for c in required if c not in df.columns]
report["null_cells"] = int(df[required].isnull().sum().sum())
report["dup_stimulus_ids"] = int(df["stimulus_id"].duplicated().sum())
arm_counts = df.groupby("scenario_id")["arm"].agg(lambda s: tuple(sorted(s)))
report["scenarios_without_5_arms"] = int((arm_counts != ("neutral","persona","self","source","third")).sum())
report["word_count_mismatch"] = int((df["word_count"] != df["text"].str.split().str.len()).sum())

# --- 2. Split integrity ---------------------------------------------------
topic_splits = df.groupby("topic")["split"].nunique()
report["topics_in_multiple_splits"] = int((topic_splits > 1).sum())
report["split_counts"] = df.groupby("split")["scenario_id"].nunique().to_dict()
report["arms_per_split"] = {f"{a}:{b}": int(v) for (a, b), v in df.groupby(["split","arm"]).size().items()}

# --- 3. Gate re-verification on shipped rows -------------------------------
from emotion.qa import g1_emotion_word_absent, g2_arm_markers
from emotion.cards import ScenarioCard
g1_fail, g2_fail = [], []
for _, r in df[df.arm != "source"].iterrows():
    card = ScenarioCard(r["scenario_id"], r["emotion"], r["topic"], int(r["source_row"]), "", r["split"], "")
    if not g1_emotion_word_absent(r["text"], r["emotion"]):
        g1_fail.append(r["stimulus_id"])
    # g2 needs the character name; recompute it the same deterministic way
    from emotion.cards import character_name
    card = ScenarioCard(r["scenario_id"], r["emotion"], r["topic"], int(r["source_row"]), "", r["split"], character_name(r["scenario_id"]))
    if not g2_arm_markers(r["text"], r["arm"], card):
        g2_fail.append(r["stimulus_id"])
report["g1_failures_shipped"] = g1_fail
report["g2_failures_shipped"] = g2_fail

# --- 4. Text hygiene -------------------------------------------------------
ends = df["text"].str.strip().str[-1]
report["texts_not_ending_sentence"] = df[~ends.isin([".", "!", "?", '"', "”", "’"])][["stimulus_id"]].to_dict("records")[:10]
report["n_texts_not_ending_sentence"] = int((~ends.isin([".", "!", "?", '"', "”", "’"])).sum())
report["prompt_artifact_leaks"] = int(df["text"].str.contains("Rules:|Story:|Topic:|STRICT FORMAT", regex=True).sum())
nonascii = df["text"].apply(lambda t: sum(1 for ch in t if ord(ch) > 127) / max(len(t),1))
report["high_nonascii_rows"] = df[nonascii > 0.05]["stimulus_id"].tolist()

# --- 5. Distributions -------------------------------------------------------
report["rows_per_emotion"] = df.groupby("emotion")["scenario_id"].nunique().to_dict()
report["wordcount_by_arm"] = df.groupby("arm")["word_count"].describe()[["mean","std","min","max"]].round(1).to_dict()
report["n_scenarios"] = int(df["scenario_id"].nunique())
report["n_rows"] = len(df)

print(json.dumps(report, indent=1, default=str))
