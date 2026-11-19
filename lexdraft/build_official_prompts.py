"""Official DFlash evaluation prompts (utils.load_and_process_dataset formats): GSM8K test, MATH-500, HumanEval."""
import glob

import pandas as pd

from common import ROOT, jdump

SUF = "\nPlease reason step by step, and put your final answer within " + chr(92) + "boxed{}."


def first(pattern):
    fs = sorted(glob.glob(pattern, recursive=True))
    if not fs:
        raise FileNotFoundError(pattern)
    return fs[0]


gs = pd.read_parquet(first(f"{ROOT}/data/gsm8k/**/test-*.parquet"))
ma = pd.read_json(first(f"{ROOT}/data/MATH-500/**/test.jsonl"), lines=True)
he = pd.read_parquet(first(f"{ROOT}/data/openai_humaneval/**/test-*.parquet"))
rows = [{"id": f"gsm8k-{i}", "task": "gsm8k", "lang": "en", "text": q + SUF} for i, q in enumerate(gs.question)]
rows_m = [{"id": f"math500-{i}", "task": "math500", "lang": "en", "text": q + SUF} for i, q in enumerate(ma.problem)]
fmt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{}\n```"
rows_h = [{"id": f"humaneval-{i}", "task": "humaneval", "lang": "en", "text": fmt.format(p)} for i, p in enumerate(he.prompt)]
jdump(rows + rows_m, f"{ROOT}/work/prompts_official.jsonl")          # Qwen3 protocol (Table 1)
jdump(rows + rows_h, f"{ROOT}/work/prompts_official_llama.jsonl")    # LLaMA-3.1 protocol (Table 5)
jdump(rows[:100], f"{ROOT}/work/prompts_align_ar.jsonl")             # AR-SD alignment subset
print(len(rows), len(rows_m), len(rows_h))
