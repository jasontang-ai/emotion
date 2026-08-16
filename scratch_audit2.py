
import json
import pandas as pd

df = pd.read_parquet("data/out/pces_v0_1.parquet")
judged = pd.read_json("data/out/judged.jsonl", lines=True)

# 1. truncated / short rows
ends = df["text"].str.strip().str[-1]
odd = df[~ends.isin([".", "!", "?", chr(34), chr(0x201d), chr(0x2019)])]
for _, r in odd.iterrows():
    print("TRUNC:", r["stimulus_id"], "|", r["text"][-70:])
short = df[df["word_count"] < 90]
print("SHORT:", short[["stimulus_id","word_count"]].to_dict("records"))
print()

# 2. judge valence by emotion x arm (mean) — do arms preserve the emotion's sign?
judged["emotion"] = judged["emotion"].astype(str)
piv = judged[judged.arm != "neutral"].pivot_table(
    index="emotion", columns="arm", values="judge_valence", aggfunc="mean").round(2)
piv["neutral_mean"] = judged[judged.arm=="neutral"].groupby("emotion")["judge_valence"].mean().round(2)
print(piv)
print()

# 3. judge valence spread per arm — persona arm analog quality
print("judge valence std by arm:", judged.groupby("arm")["judge_valence"].std().round(2).to_dict())
print("judge arousal mean by arm:", judged.groupby("arm")["judge_arousal"].mean().round(2).to_dict())
