
import asyncio, json
from emotion.dataset import build_rows
from emotion.cards import sample_cards
from emotion.generate import DEFAULT_MODEL, generate_batch
from emotion.qa import judge_prompt, parse_judge, lexicon_valence

EXCLUDED = {"afraid-315dc175","angry-784bb398","angry-f1c8802f","desperate-784bb398"}
cards = [c for c in sample_cards("data/source/stories.parquet") if c.scenario_id not in EXCLUDED]
texts = json.loads(open("data/gen/texts.json").read())
rows = build_rows(cards, texts, DEFAULT_MODEL)
row_of = {r.stimulus_id: r for r in rows}

# changed rows = all persona arms + the non-persona truncation fixes
changed = [r for r in rows if r.arm == "persona"
           or r.stimulus_id in {"sad-9e7897f8:third", "sad-449ff49b:third"}]
print("re-judging", len(changed), "changed rows")

prompts = [judge_prompt(r.text) for r in changed]
results = asyncio.run(generate_batch(prompts, model=DEFAULT_MODEL, concurrency=16, thinking=True))
new_ratings = {}
for row, res in zip(changed, results, strict=True):
    try:
        v, a = parse_judge(res.text)
    except ValueError:
        v, a = None, None
    new_ratings[row.stimulus_id] = {
        "judge_valence": v, "judge_arousal": a,
        "lexicon_compound": round(lexicon_valence(row.text), 3),
    }

judged = [json.loads(l) for l in open("data/out/judged.jsonl")]
merged = 0
for j in judged:
    sid = j["stimulus_id"]
    if sid in new_ratings:
        j.update(new_ratings[sid])
        j["rejudged_v0_2"] = True
        merged += 1
    # sanity: text in judged must match current dataset text
    assert j["text"] == row_of[sid].text or sid in new_ratings, f"stale text: {sid}"
with open("data/out/judged.jsonl", "w") as f:
    f.write("\n".join(json.dumps(j, sort_keys=True) for j in judged) + "\n")
print("merged", merged, "re-judged rows; unparseable:", sum(1 for v in new_ratings.values() if v["judge_valence"] is None))
