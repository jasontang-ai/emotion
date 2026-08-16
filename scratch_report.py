
import json
import pandas as pd
jdf = pd.read_json("data/out/judged.jsonl", lines=True)
POS = {"joyful","grateful","proud"}; NEG = {"afraid","angry","sad","ashamed","desperate"}
core = jdf[(jdf.arm!="neutral") & (jdf.emotion.isin(POS|NEG)) & jdf.judge_valence.notna()]
ok = ((core.emotion.isin(NEG)) & (core.judge_valence<0)) | ((core.emotion.isin(POS)) & (core.judge_valence>0))
anchor7 = ok.mean()
calm = jdf[(jdf.emotion=="calm") & (jdf.arm!="neutral") & jdf.judge_arousal.notna()]
sur = jdf[(jdf.emotion=="surprised") & (jdf.arm!="neutral") & jdf.judge_arousal.notna()]
neu = jdf[(jdf.arm=="neutral") & jdf.judge_valence.notna()]
pairs = jdf[jdf.arm.isin(["self","third"])].pivot_table(index="scenario_id", columns="arm",
                                                        values="judge_valence", aggfunc="first").dropna()
match = (abs(pairs["self"]-pairs["third"])<=1).mean()
# persona sign check (the v0.2 fix)
per = jdf[(jdf.arm=="persona") & jdf.emotion.isin(POS|NEG) & jdf.judge_valence.notna()]
pok = ((per.emotion.isin(NEG)) & (per.judge_valence<0)) | ((per.emotion.isin(POS)) & (per.judge_valence>0))
report = {
 "version": "pces.v0.2",
 "n_judged": len(jdf),
 "unparseable": int(jdf.judge_valence.isna().sum()),
 "h1_sign_agreement_valence_anchor7": round(float(anchor7),3),
 "h1_calm_low_arousal": round(float((calm.judge_arousal<=1).mean()),3),
 "h1_surprised_high_arousal": round(float((sur.judge_arousal>=2).mean()),3),
 "h1_neutral_flat_judge": round(float((neu.judge_valence==0).mean()),3),
 "h3_self_third_valence_match": round(float(match),3),
 "v0_2_persona_sign_agreement": round(float(pok.mean()),3),
 "judge_model": "deepseek/deepseek-v4-flash-0731 (reasoning enabled)",
 "note": "v0.2: persona arm regenerated with valence-neutral frame; only changed rows re-judged (248, flagged rejudged_v0_2).",
}
open("data/out/validity_report.json","w").write(json.dumps(report, indent=1, sort_keys=True))
print(json.dumps(report, indent=1))
