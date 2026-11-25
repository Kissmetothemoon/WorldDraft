"""Extra Aya chat prompts for conditions that lost most samples to looping / small n (not used in the main grid)."""
import glob
import random

import pandas as pd

from common import LANGS, ROOT, jdump, jload

used = {r["text"] for r in jload(f"{ROOT}/work/prompts_eval.jsonl")}
fs = [f for f in glob.glob(f"{ROOT}/data/aya_dataset/**/*.parquet", recursive=True) if "/data/train" in f]
aya = pd.concat([pd.read_parquet(f) for f in fs])
rows = []
for lang in ["am", "sw", "te", "hi", "bn", "ar", "ko", "th"]:
    sub = aya[(aya.language == LANGS[lang][2]) & aya.inputs.str.len().between(20, 600) & ~aya.inputs.isin(used)]
    idx = list(sub.index.unique())
    random.Random(1).shuffle(idx)
    for i, (_, r) in enumerate(sub.loc[idx[:300]].iterrows()):
        rows.append({"id": f"chatx-{lang}-{i}", "task": "chat", "lang": lang, "text": r["inputs"]})
jdump(rows, f"{ROOT}/work/prompts_extra.jsonl")
print(len(rows))
