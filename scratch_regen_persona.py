
import asyncio, json
from pathlib import Path
from emotion.arms import PROMPTS
from emotion.cards import sample_cards
from emotion.generate import DEFAULT_MODEL, generate_batch

cards = sample_cards("data/source/stories.parquet")
ckpt = Path("data/gen/texts.json")
texts = json.loads(ckpt.read_text())

# regenerate all persona arms with the neutral frame + the 6 defective rows
trunc = ["proud-1295a29c:persona","surprised-d37134c0:persona","sad-9e7897f8:third",
         "sad-449ff49b:third","ashamed-e11a4f15:persona"]
todo, keys = [], []
for c in cards:
    key = f"{c.scenario_id}:persona"
    todo.append(PROMPTS["persona"](c)); keys.append(key)
for key in trunc:
    sid, arm = key.rsplit(":", 1)
    if key in keys:   # persona truncations already queued
        continue
    c = next(c for c in cards if c.scenario_id == sid)
    todo.append(PROMPTS[arm](c)); keys.append(key)

print(f"regenerating {len(todo)} stimuli")
results = asyncio.run(generate_batch(todo, model=DEFAULT_MODEL, temperature=0.5))
for key, result in zip(keys, results, strict=True):
    texts[key] = result.text
ckpt.write_text(json.dumps(texts))
print("done")
