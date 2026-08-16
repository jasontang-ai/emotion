
import asyncio, json
from emotion.cards import sample_cards
from emotion.dataset import build_rows, write_dataset
from emotion.generate import DEFAULT_MODEL, generate_batch
from emotion.qa import g1_emotion_word_absent, g2_arm_markers, judge_prompt, parse_judge, lexicon_valence

cards = sample_cards("data/source/stories.parquet")
texts = json.loads(open("data/gen/texts.json").read())
kept = [c for c in cards if c.scenario_id not in
        {"afraid-315dc175","angry-784bb398","angry-f1c8802f","desperate-784bb398"}]

# deterministic gates on new persona arm
card_of = {c.scenario_id: c for c in kept}
bad = [f"{c.scenario_id}:persona" for c in kept
       if not (g1_emotion_word_absent(texts[f"{c.scenario_id}:persona"], c.emotion)
               and g2_arm_markers(texts[f"{c.scenario_id}:persona"], "persona", c))]
print("persona gate failures:", bad)

rows = build_rows(kept, texts, DEFAULT_MODEL)
manifest = write_dataset(rows, "data/out", {"gate_report": {
    "v0_2": "persona arm regenerated with valence-neutral frame; 6 truncated/short rows regenerated",
    "persona_gate_failures": bad,
    "excluded_scenarios": ["afraid-315dc175","angry-784bb398","angry-f1c8802f","desperate-784bb398"]}})

# re-judge everything (thinking on)
records = [r for r in rows]
prompts = [judge_prompt(r.text) for r in records]
results = asyncio.run(generate_batch(prompts, model=DEFAULT_MODEL, concurrency=16, thinking=True))
judged = []
for row, res in zip(records, results, strict=True):
    try:
        v, a = parse_judge(res.text)
    except ValueError:
        v, a = None, None
    d = row.__dict__ if hasattr(row, "__dict__") else dict(row)
    judged.append({**d, "judge_valence": v, "judge_arousal": a,
                   "lexicon_compound": round(lexicon_valence(row.text), 3)})
with open("data/out/judged.jsonl", "w") as f:
    f.write("\n".join(json.dumps(j, sort_keys=True) for j in judged) + "\n")
print("rejudged", len(judged), "| unparseable:", sum(1 for j in judged if j["judge_valence"] is None))
