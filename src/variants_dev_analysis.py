"""Variants development analysis (exploratory, descriptive; docs/THEORY_ADDENDUM_ZH.md §6–§7; reviewer batch 7; user
follow-up). Written and committed BEFORE the variants run starts. Same 7 unit tasks x 20 frozen lines, Qwen2.5-1.5B,
16k, budget 512, B32 / U32 as P-grid. Arms: cobs, cobs+g2f (from runs/ruler/pgrid_pred/b32) and, from
runs/ruler/variants_pred: cobs+nb, cobs+nbr, cobs+nbh (neighbour-read alternatives; no extra summary memory), mean,
mean+g2f (MoBA-style mean pooling) and quest (B32). A sample enters only if every arm has it (fixed rule).
Units: one per answer reference (correct iff it occurs in the post-processed prediction); 32-grid class in0 / repair /
out / fixed from cluster/unit_lengths.py v3 (U = 32).
Registered expectations (development, no decision rule):
  E1 G ordering: G_mean >= G_cobs (and both > G_quest), G_s = acc(in0) - acc(repair) of the single-grid arm s;
  E2 L1' duality: cobs+nbh close to cobs+g2f on single-unit tasks (niah_single_1/2, niah_multikey_1/2), worse on
     multi-unit tasks (niah_multivalue, niah_multiquery, vt), because nbh reads up to 2B per seed (about half the seeds);
  E3 equal summary memory: mean+g2f vs quest (both 8 floats per token per KV head) — reported, no direction claimed.
Reports per arm: unit accuracy overall / per task / single- vs multi-unit tasks / by 32-grid class; G and R per scorer
(R = (acc'(repair) - acc(repair)) / G for cobs and mean); differences with a sample-level bootstrap within task (2000):
G_mean - G_cobs, nbh - g2f (single, multi), nb - g2f, nbr - g2f, mean+g2f - quest. Unified table columns: summary
floats per token (results/summary_memory.log) and read per seed.
Usage: variants_dev_analysis.py DATA_DIR PGRID_B32_DIR VARIANTS_DIR TOKENIZER [N]"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cluster"))
from aggregate import postprocess_pred  # noqa: E402
from unit_lengths import measure  # noqa: E402

TASKS = ("niah_single_1", "niah_single_2", "niah_multikey_1", "niah_multikey_2", "niah_multivalue", "niah_multiquery",
         "vt")
SINGLE = ("niah_single_1", "niah_single_2", "niah_multikey_1", "niah_multikey_2")
ARMS = ("cobs", "cobs+g2f", "cobs+nb", "cobs+nbr", "cobs+nbh", "mean", "mean+g2f", "quest")
MEM = {"cobs": "5.12", "cobs+g2f": "10.25", "cobs+nb": "5.12", "cobs+nbr": "5.12", "cobs+nbh": "5.12", "mean": "4.00",
       "mean+g2f": "8.00", "quest": "8.00"}
READ = {"cobs": "B", "cobs+g2f": "B", "cobs+nb": "2B (half seeds)", "cobs+nbr": "2B (half seeds)",
        "cobs+nbh": "<=2B", "mean": "B", "mean+g2f": "B", "quest": "B"}


def load(path):
    return {json.loads(l)["sample_uid"]: json.loads(l) for l in open(path)} if os.path.exists(path) else {}


def main(data, pg, vd, tokdir, n):
    u32 = measure(data, tokdir, n, 32, TASKS, 64, 32)
    recs = []
    for t in TASKS:
        lines = [json.loads(l) for l in open(os.path.join(data, t, "validation.jsonl"))][:n]
        src = {a: pg if a in ("cobs", "cobs+g2f") else f"{vd}/nb" if "+nb" in a else f"{vd}/sum" for a in ARMS}
        rows = {a: load(f"{src[a]}/{a}@512/{t}.jsonl") for a in ARMS}
        for i, ln in enumerate(lines):
            uid = next((u for u in rows["cobs"] if u.endswith(f":{t}:{i}")), None)
            if uid is None or any(uid not in rows[a] for a in ARMS):
                continue
            cls = u32["tasks"][t]["samples"][i]["cls"]
            for k, ref in enumerate(ln["outputs"]):
                recs.append(dict(task=t, i=i, cls=cls[k],
                                 ok={a: int(ref.lower() in postprocess_pred(rows[a][uid]["pred"]).lower()) for a in ARMS}))
    samples = sorted({(r["task"], r["i"]) for r in recs})
    print(f"variants development (exploratory): {len(recs)} units, {len(samples)} samples "
          f"(of {len(TASKS) * n}; samples lacking any arm are excluded by the fixed rule)")
    by = {s: [r for r in recs if (r["task"], r["i"]) == s] for s in samples}
    rng = np.random.default_rng(0)
    picks = []
    for _ in range(2000):
        p = []
        for t in TASKS:
            ss = [s for s in samples if s[0] == t]
            p += [ss[j] for j in rng.integers(0, len(ss), len(ss))] if ss else []
        picks.append(p)

    def acc(u, a):
        return np.mean([r["ok"][a] for r in u]) if u else np.nan

    def G(u, a):
        return acc([r for r in u if r["cls"] == "in0"], a) - acc([r for r in u if r["cls"] == "repair"], a)

    def boot(stat):
        v = [stat([r for s in p for r in by[s]]) for p in picks]
        v = [x for x in v if np.isfinite(x)]
        return np.percentile(v, [2.5, 97.5]) if v else (np.nan, np.nan)

    print(f"{'arm':10s} {'mem/tok':>7s} {'read/seed':>15s} {'all':>6s} {'single':>6s} {'multi':>6s} "
          f"{'in0':>6s} {'repair':>6s} {'out':>6s}")
    for a in ARMS:
        cl = {c: acc([r for r in recs if r["cls"] == c], a) for c in ("in0", "repair", "out")}
        print(f"{a:10s} {MEM[a]:>7s} {READ[a]:>15s} {acc(recs, a):6.3f} "
              f"{acc([r for r in recs if r['task'] in SINGLE], a):6.3f} "
              f"{acc([r for r in recs if r['task'] not in SINGLE], a):6.3f} "
              + " ".join(f"{cl[c]:6.3f}" for c in ("in0", "repair", "out")))
    print("per task:")
    for t in TASKS:
        u = [r for r in recs if r["task"] == t]
        print(f"  {t:16s} units {len(u):4d}: " + "  ".join(f"{a} {acc(u, a):.3f}" for a in ARMS))
    print("E1 G ordering (G = acc(in0) - acc(repair), single-grid arm; R for +g2f):")
    for s, tw in (("cobs", "cobs+g2f"), ("mean", "mean+g2f"), ("quest", None)):
        g = G(recs, s)
        rep = [r for r in recs if r["cls"] == "repair"]
        R = (acc(rep, tw) - acc(rep, s)) / g if tw and g > 0.05 else np.nan
        lo, hi = boot(lambda u, s=s: G(u, s))
        print(f"  G_{s:6s} {g:+.3f} [{lo:+.3f}, {hi:+.3f}]  R {R:.2f}  repair units {len(rep)}")
    lo, hi = boot(lambda u: G(u, "mean") - G(u, "cobs"))
    print(f"  G_mean - G_cobs {G(recs, 'mean') - G(recs, 'cobs'):+.3f} [{lo:+.3f}, {hi:+.3f}]")

    def diff(name, a, b, filt=lambda r: True):
        u = [r for r in recs if filt(r)]
        lo, hi = boot(lambda v: acc([r for r in v if filt(r)], a) - acc([r for r in v if filt(r)], b))
        print(f"  {name:34s} units {len(u):4d}: {acc(u, a) - acc(u, b):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("E2 / alternatives (difference in unit accuracy):")
    diff("cobs+nbh - cobs+g2f (single-unit)", "cobs+nbh", "cobs+g2f", lambda r: r["task"] in SINGLE)
    diff("cobs+nbh - cobs+g2f (multi-unit)", "cobs+nbh", "cobs+g2f", lambda r: r["task"] not in SINGLE)
    diff("cobs+nb - cobs+g2f", "cobs+nb", "cobs+g2f")
    diff("cobs+nbr - cobs+g2f", "cobs+nbr", "cobs+g2f")
    diff("cobs+nbh - cobs", "cobs+nbh", "cobs")
    print("E3 equal summary memory (8 floats/token):")
    diff("mean+g2f - quest", "mean+g2f", "quest")
    diff("mean+g2f - mean", "mean+g2f", "mean")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 20)
