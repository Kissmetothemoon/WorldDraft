"""Monolingual lexicon text from Wikipedia (20231101 dump; set HF_ENDPOINT for a mirror), tokenised with the target tokenizer.

usage: wiki_lex_prep.py <tokenizer> <out_dir> <budget_tokens_per_lang> [langs]
writes <out_dir>/wiki_<lang>.jsonl rows {"lang", "output_ids"} (paragraph chunks), cumulative up to the budget.
sr-Latn is produced by transliterating the Serbian (Cyrillic) Wikipedia; hi-Latn has no source and is skipped.
"""
import json
import os
import sys
from multiprocessing import Pool

WIKI = {"en": "en", "de": "de", "fr": "fr", "es": "es", "ru": "ru", "zh": "zh", "ja": "ja", "ko": "ko", "ar": "ar", "th": "th",
        "sw": "sw", "hi": "hi", "bn": "bn", "te": "te", "am": "am", "sr-Cyrl": "sr", "sr-Latn": "sr"}
CYR2LAT = dict(zip("абвгдђежзијклљмнњопрстћуфхцчџшАБВГДЂЕЖЗИЈКЛЉМНЊОПРСТЋУФХЦЧЏШ",
                   ["a", "b", "v", "g", "d", "đ", "e", "ž", "z", "i", "j", "k", "l", "lj", "m", "n", "nj", "o", "p", "r", "s", "t", "ć",
                    "u", "f", "h", "c", "č", "dž", "š", "A", "B", "V", "G", "D", "Đ", "E", "Ž", "Z", "I", "J", "K", "L", "Lj", "M", "N",
                    "Nj", "O", "P", "R", "S", "T", "Ć", "U", "F", "H", "C", "Č", "Dž", "Š"]))
tokp, out_dir, budget = sys.argv[1], sys.argv[2], int(float(sys.argv[3]))
langs = sys.argv[4].split(",") if len(sys.argv) > 4 else list(WIKI)
os.makedirs(out_dir, exist_ok=True)


def run(lang):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokp)
    code = WIKI[lang]
    files = sorted(f for f in list_repo_files("wikimedia/wikipedia", repo_type="dataset") if f.startswith(f"20231101.{code}/"))
    n, rows = 0, 0
    with open(f"{out_dir}/wiki_{lang}.jsonl", "w", encoding="utf-8") as fo:
        for f in files:
            path = hf_hub_download("wikimedia/wikipedia", f, repo_type="dataset")
            tab = pq.read_table(path, columns=["text"])
            for text in tab.column("text").to_pylist():
                if lang == "sr-Latn":
                    text = "".join(CYR2LAT.get(c, c) for c in text)
                for para in text.split("\n"):
                    para = para.strip()
                    if len(para) < 40:
                        continue
                    ids = tok.encode(para, add_special_tokens=False)[:1024]
                    fo.write(json.dumps({"lang": lang, "output_ids": ids}) + "\n")
                    n += len(ids)
                    rows += 1
                    if n >= budget:
                        return lang, n, rows
    return lang, n, rows


if __name__ == "__main__":
    with Pool(len(langs)) as p:
        for lang, n, rows in p.imap_unordered(run, langs):
            print(lang, n, "tokens", rows, "paragraphs", flush=True)
