"""Build the evaluation prompt grid (language x task) for stage 0/1."""
import glob
import random

import pandas as pd

from common import LANGS, ROOT, jdump

random.seed(0)
rows = []

# MGSM: native-language math questions
for lang, (_, _, _, mg) in LANGS.items():
    if mg is None:
        continue
    df = pd.read_parquet(f"{ROOT}/data/mgsm/{mg}/test-00000-of-00001.parquet")
    for i, q in enumerate(df["question"].tolist()):
        rows.append({"id": f"math-{lang}-{i}", "task": "math", "lang": lang, "text": q})

# Translation: English MGSM questions into every non-English condition; German -> English for en
en_q = pd.read_parquet(f"{ROOT}/data/mgsm/en/test-00000-of-00001.parquet")["question"].tolist()
de_q = pd.read_parquet(f"{ROOT}/data/mgsm/de/test-00000-of-00001.parquet")["question"].tolist()
for lang in LANGS:
    src = de_q if lang == "en" else en_q
    for i, q in enumerate(src):
        rows.append({"id": f"translate-{lang}-{i}", "task": "translate", "lang": lang, "text": q})

# Chat: native Aya prompts (held-out ids recorded so training data can exclude them)
fs = [f for f in glob.glob(f"{ROOT}/data/aya_dataset/**/*.parquet", recursive=True) if "/data/train" in f]
aya = pd.concat([pd.read_parquet(f) for f in fs])
used = []
for lang, (_, _, aya_name, _) in LANGS.items():
    if aya_name is None:
        continue
    sub = aya[(aya.language == aya_name) & aya.inputs.str.len().between(20, 600)]
    idx = list(sub.index.unique())
    random.shuffle(idx)
    sub = sub.loc[idx[:200]]
    for i, (k, r) in enumerate(sub.iterrows()):
        rows.append({"id": f"chat-{lang}-{i}", "task": "chat", "lang": lang, "text": r["inputs"]})
        used.append(r["inputs"])

jdump(rows, f"{ROOT}/work/prompts_eval.jsonl")
jdump([{"text": t} for t in used], f"{ROOT}/work/aya_eval_heldout.jsonl")
from collections import Counter
print(len(rows), Counter((r["task"]) for r in rows))
