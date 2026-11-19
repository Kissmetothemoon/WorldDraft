"""Exact decomposition of engine speedup with untimed wall clock: speedup = (steps_AR / steps_SD) x (c_AR / c_SD),
steps from the (deterministic, greedy) timed pass on the same prompts, c = untimed time / steps (prof-n prompts only)."""
import pickle, sys
import numpy as np
D = sys.argv[1]
LANGS = ["en", "de", "zh", "ja", "ko", "th", "hi", "bn", "te", "sr-Cyrl"]
ORDER = ["draft", "dpub", "dpub_v3", "dml", "dml_v7", "dml_v3", "dml_dyn"]
LABEL = {"draft": "AR 草稿 0.6B 多语言（k=5）", "dpub": "公开 DFlash（验证 15）", "dpub_v3": "公开 DFlash（验证 3）", "dml": "多语言 DFlash（验证 15）",
         "dml_v7": "多语言 DFlash（验证 7）", "dml_v3": "多语言 DFlash（验证 3）", "dml_dyn": "多语言 DFlash（按并发 15/7/3）"}
R = {n: pickle.load(open(f"{D}/{n}.pkl", "rb")) for n in ["ar"] + ORDER}
def stat(n, l, c):
    rs = [r for r in R[n]["res"] if r["lang"] == l and r["conc"] == c]
    # untimed time restricted to the first prof-n prompts is not stored separately -> scale untimed time by token share
    P = R[n]["args"]["prof_n"]
    steps = sum(len(r["prof"]["steps"]) for r in rs)
    t_un = sum(r["time"] * (r["prof"]["tokens"] / r["tokens"]) for r in rs)  # untimed time attributable to the profiled prompts
    tok = sum(r["prof"]["tokens"] for r in rs)
    return steps, t_un, tok
L = ["| 方法 | 并发 | 步数比 AR/SD | 每步成本比 c_SD/c_AR（不计时运行） | 加速比 = 两者之商 |", "|---|---|---|---|---|"]
per = {}
for n in ORDER:
    for c in (1, 8, 32):
        a = [stat("ar", l, c) for l in LANGS]; m = [stat(n, l, c) for l in LANGS]
        sr = np.mean([x[0] / y[0] for x, y in zip(a, m)])
        cr = np.mean([(y[1] / y[0]) / (x[1] / x[0]) for x, y in zip(a, m)])
        sp = np.mean([(x[1] / x[2]) / (y[1] / y[2]) for x, y in zip(a, m)])
        per[(n, c)] = [((x[0] / y[0]), (y[1] / y[0]) / (x[1] / x[0])) for x, y in zip(a, m)]
        L.append(f"| {LABEL[n]} | {c} | {sr:.2f} | {cr:.2f} | {sp:.2f} |")
L += ["", "并发 32，各语言：步数比 / 每步成本比", "", "| 语言 | " + " | ".join(LABEL[n] for n in ORDER) + " |", "|---|" + "---|" * len(ORDER)]
for i, l in enumerate(LANGS):
    L.append(f"| {l} | " + " | ".join(f"{per[(n,32)][i][0]:.2f} / {per[(n,32)][i][1]:.2f}" for n in ORDER) + " |")
L += ["", "AR 基线每步耗时（ms，不计时运行）：" + "，".join(f"并发 {c}: {1e3*np.mean([stat('ar',l,c)[1]/stat('ar',l,c)[0] for l in LANGS]):.1f}" for c in (1, 8, 32))]
txt = "\n".join(L) + "\n"; open(f"{D}/prof_exact.md", "w", encoding="utf-8").write(txt); print(txt)
