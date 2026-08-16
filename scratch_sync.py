
import json
import pandas as pd
df = pd.read_parquet("data/out/pces_v0_1.parquet")
text_of = dict(zip(df["stimulus_id"], df["text"]))
judged = [json.loads(l) for l in open("data/out/judged.jsonl")]
fixed = 0
for j in judged:
    cur = text_of.get(j["stimulus_id"])
    if cur is not None and j["text"] != cur:
        j["text"] = cur
        fixed += 1
with open("data/out/judged.jsonl", "w") as f:
    f.write("\n".join(json.dumps(x, sort_keys=True) for x in judged) + "\n")
print("texts synced:", fixed)

# verify the persona valence fix
jdf = pd.DataFrame(judged)
piv = jdf[jdf.arm != "neutral"].pivot_table(index="emotion", columns="arm",
                                            values="judge_valence", aggfunc="mean").round(2)
print(piv[["persona","self","third"]])
