"""Baseline-alignment gate. Exits non-zero (so the scheduler skips every dependent job) if any checked gate fails.

usage: check_alignment.py G1 G2 ...   (gates available on this server)
G1  DFlash + Qwen3-4B, official protocol     vs DFlash Table 1 (T=0): GSM8K tau 6.53, MATH-500 tau 7.84
G2  DFlash + LLaMA-3.1-8B, official protocol  vs DFlash Table 5:       GSM8K tau 4.32, HumanEval tau 4.91
G3  our AR-draft SD vs HF assisted generation (Qwen3-8B + Qwen3-0.6B, K=5)
"""
import glob
import sys

import numpy as np

from common import ROOT, pload

TAU_TOL = 0.06
PAPER = {"G1": ("official_q4", {"gsm8k": 6.53, "math500": 7.84}),
         "G2": ("official_ll8", {"gsm8k": 4.32, "humaneval": 4.91})}
ok_all = True
lines = ["# Baseline alignment gate\n"]
for g in sys.argv[1:]:
    if g in PAPER:
        tag, ref = PAPER[g]
        sp = sum((pload(f) for f in sorted(glob.glob(f"{ROOT}/work/{tag}/speed_*.pkl"))), [])
        lines.append(f"## {g} ({tag}, n={len(sp)})\n")
        lines.append("| benchmark | n | our tau | paper tau | rel. diff | our speedup (NVIDIA L20, HF) | pass |")
        lines.append("|---|---|---|---|---|---|---|")
        for task, pt in ref.items():
            xs = [x for x in sp if x["task"] == task]
            if not xs:
                lines.append(f"| {task} | 0 | - | {pt} | - | - | NO DATA |")
                ok_all = False
                continue
            tau = sum(sum(x["accs"]) for x in xs) / sum(len(x["accs"]) for x in xs)
            spd = (sum(x["n_sp"] for x in xs) / sum(x["t_sp"] for x in xs)) / (sum(x["n_ar"] for x in xs) / sum(x["t_ar"] for x in xs))
            rel = tau / pt - 1
            ok = abs(rel) <= TAU_TOL
            ok_all &= ok
            lines.append(f"| {task} | {len(xs)} | {tau:.2f} | {pt:.2f} | {rel:+.1%} | {spd:.2f}x | {'PASS' if ok else 'FAIL'} |")
        lines.append("")
    elif g == "G3":
        rep = sum((pload(f) for f in sorted(glob.glob(f"{ROOT}/work/align_ar/align_*.pkl"))), [])
        if not rep:
            lines.append("## G3: NO DATA\n")
            ok_all = False
            continue
        tps = {m: sum(r[m]["n"] for r in rep) / sum(r[m]["t"] for r in rep) for m in ("ar", "hf_assisted", "ours")}
        ident = {m: np.mean([r[m]["ids"] == r["ar"]["ids"] for r in rep]) for m in ("hf_assisted", "ours")}
        tau_real = sum(sum(r["accs"]) for r in rep) / sum(len(r["accs"]) for r in rep)
        tau_off = sum(sum(r["offline_accs"]) for r in rep) / sum(len(r["offline_accs"]) for r in rep)
        c_speed = tps["ours"] >= 0.9 * tps["hf_assisted"]
        c_ident = ident["ours"] >= ident["hf_assisted"] - 0.05
        c_tau = abs(tau_real / tau_off - 1) <= 0.02
        ok_all &= c_speed and c_ident and c_tau
        lines += [f"## G3 (n={len(rep)})\n", "| check | value | pass |", "|---|---|---|",
                  f"| tokens/s: AR / HF assisted / ours | {tps['ar']:.1f} / {tps['hf_assisted']:.1f} / {tps['ours']:.1f} | {'PASS' if c_speed else 'FAIL'} (ours >= 0.9 x HF) |",
                  f"| outputs identical to AR: HF assisted / ours | {ident['hf_assisted']:.2f} / {ident['ours']:.2f} | {'PASS' if c_ident else 'FAIL'} |",
                  f"| tau real loop / offline recomputation | {tau_real:.3f} / {tau_off:.3f} | {'PASS' if c_tau else 'FAIL'} (within 2%) |", ""]
for g in sys.argv[1:]:
    if g not in ("G4", "G4b"):
        continue
    # G4a: engine (vLLM) DFlash tau vs paper; G4b: engine draft_model tau vs our validated HF loop (G3)
    lines += ["## G4 optimised engine (vLLM)\n", "| check | ours | reference | rel. diff | pass |", "|---|---|---|---|---|"]
    for task, pt in ((("gsm8k", 6.54), ("math500", 7.87)) if g == "G4" else ()):
        fs = glob.glob(f"{ROOT}/work/eng_official/dflash_{task}.pkl")
        if not fs:
            lines.append(f"| DFlash {task} tau | - | {pt} | - | NO DATA |")
            ok_all = False
            continue
        rs = [r for r in pload(fs[0])["res"] if r["task"] == task]
        tau = float(np.mean([r["tau"] for r in rs]))
        ok = abs(tau / pt - 1) <= TAU_TOL
        ok_all &= ok
        lines.append(f"| DFlash {task} tau (engine) | {tau:.2f} | {pt:.2f} (paper) | {tau/pt-1:+.1%} | {'PASS' if ok else 'FAIL'} |")
    fs = glob.glob(f"{ROOT}/work/eng_align/draft.pkl")
    rep = sum((pload(f) for f in sorted(glob.glob(f"{ROOT}/work/align_ar/align_*.pkl"))), [])
    if fs and rep:
        tau_e = float(np.mean([r["tau"] for r in pload(fs[0])["res"]]))
        tau_h = sum(sum(r["accs"]) for r in rep) / sum(len(r["accs"]) for r in rep)
        ok = abs(tau_e / tau_h - 1) <= 0.03
        ok_all &= ok
        lines.append(f"| draft_model tau (engine, Qwen3-0.6B, K=5) | {tau_e:.3f} | {tau_h:.3f} (our HF loop, G3) | {tau_e/tau_h-1:+.1%} | {'PASS' if ok else 'FAIL'} |")
    else:
        lines.append("| draft_model tau | - | - | - | NO DATA |")
        ok_all = False
    lines.append("")
lines.append(f"**Overall: {'PASS' if ok_all else 'FAIL'}**")
out = "\n".join(lines) + "\n"
open(f"{ROOT}/work/alignment_{'_'.join(sys.argv[1:])}.md", "w").write(out)
print(out)
sys.exit(0 if ok_all else 1)
