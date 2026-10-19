"""rep2 — replication of the round-1 development conclusions on fresh samples (reviewer batch 12; user: "吃透、验证、
高要求不变"). Written, with its decision rules, and committed BEFORE any rep2 prediction exists. Development /
descriptive only: every rule below is judged on its own (no family-wise error control) and nothing here enters the
cluster's confirmatory conclusions. The round-1 conclusions were formed AFTER looking at round-1 data; rep2 is their
first test on samples no analysis has used.
Data: runs/ruler/rep2_qwen1.5b_16k (official RULER prepare, 7 unit tasks x 100, 16k; prep_manifest sha256
ba4f5acd1bd3fc5d184b93d2a7a3e1676e2a40b6e87b6517abdcdafc4b6689f9). Lines 0..29 are byte-identical to the round-1 data
(lines 20..29 were only tokenised for unit lengths in round 1, no model output); lines 20..99 are the evaluation set
(80 per task; no duplicate input with lines 0..19, checked before any prediction). Each arm runs --limit 100: lines
0..19 give a determinism check against the round-1 outputs of the same arm (pred equality), not used in R1–R6.
Arms (budget 512, U = 32, one method per process): dense, cobs, cobs+g2f, mean, mean+g2f, quest32, quest32+g2f
(B32); cobs16, quest16 (B16). R7 (budget 1024, lines 20..59): cobs, cobs+g2f, cobs16, quest32, mean+g2f.
Unit = one per answer reference, correct iff it occurs in the RULER-post-processed prediction (case-insensitive).
Intervals: sample-level bootstrap stratified by task, 2000 resamples, numpy seed 0 (R4 also a 90% interval).
Rules (fixed now):
  R1 cobs16 - cobs+g2f (equal memory): 95% lower > 0 -> "复现"; point > 0 but CI contains 0 -> "方向一致"; else "未复现".
  R2 quest32 - mean+g2f (equal memory): 95% LOWER bound >= 0.15 -> "复现"; point >= 0.15 but lower < 0.15 -> "部分";
     else "未复现".
  R3 G_mean - G_cobs and G_cobs - G_quest (G = acc(in0) - acc(repair) of the single-grid arm, 32-grid classes):
     both 95% lower > 0 -> "完全复现"; exactly one -> "部分"; none -> "未复现".
  R4 quest32+g2f - quest16 (equal memory): 90% interval inside [-0.06, +0.06] -> "等价"; else "未判定等价"; 95% reported.
  R5 cobs+g2f - cobs and mean+g2f - mean (fixed B): each 95% lower > 0 -> "复现", else "未复现".
  R6 gain law at fixed B, cross-fitted kappa (cluster/gain_law.analyse) per scorer (cobs, mean, quest) and pooled:
     95% CI contains 1 and not 0.5 -> "相符"; excludes 1 -> "不成立"; else "不确定".
  R7 budget 1024: the point estimates of R1 (cobs16 - cobs+g2f) and R2 (quest32 - mean+g2f) have the same sign as at
     512 -> "方向一致", else "方向不一致" (direction only).
Amendment (2026-10-18, before any analysis output; found by the Codex independent audit, counterexamples.json):
  the registered R3 line computed n3 = (l1 > 0) + (l2 > 0) on numpy scalars, which is a logical OR of numpy.bool and
  then raised TypeError when used as a list index (numpy 2.5.4), so R3 would have aborted the analysis. Changed to
  int(l1 > 0) + int(l2 > 0), which is the registered rule ("both lower bounds > 0 -> 完全复现, one -> 部分"); no
  other line changed.
Usage: rep2_analysis.py [ROOT]"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, os.path.join(ROOT, "cluster"))
from aggregate import postprocess_pred  # noqa: E402
from gain_law import analyse  # noqa: E402
from unit_lengths import measure  # noqa: E402

DATA = os.path.join(ROOT, "runs/ruler/rep2_qwen1.5b_16k")
PRED = os.path.join(ROOT, "runs/ruler/rep2_pred")
ROUND1 = os.path.join(ROOT, "runs/ruler")
TOK = "/home/kooki3/models/qwen2.5-1.5b-instruct"
TASKS = ("niah_single_1", "niah_single_2", "niah_multikey_1", "niah_multikey_2", "niah_multivalue", "niah_multiquery",
         "vt")
ARMS = {"dense": "dense", "cobs": "cobs", "cobs+g2f": "cobs+g2f", "mean": "mean", "mean+g2f": "mean+g2f",
        "quest32": "quest", "quest32+g2f": "quest+g2f", "cobs16": "cobs", "quest16": "quest"}
R1_DIRS = {"cobs": "pgrid_pred/b32/cobs@512", "cobs+g2f": "pgrid_pred/b32/cobs+g2f@512", "cobs16": "pgrid_pred/b16/cobs@512",
           "mean": "variants_pred/sum/mean@512", "mean+g2f": "variants_pred/sum/mean+g2f@512",
           "quest32": "variants_pred/sum/quest@512", "quest32+g2f": "quest_pgrid/b32/quest+g2f@512",
           "quest16": "quest_pgrid/b16/quest@512"}
R7_ARMS = ("cobs", "cobs+g2f", "cobs16", "quest32", "mean+g2f")
B_BOOT, SEED = 2000, 0


def path(arm, budget):
    return os.path.join(PRED, f"b{budget}", arm, f"{ARMS[arm]}@{budget}")


def load(d, t):
    p = os.path.join(d, f"{t}.jsonl")
    return {int(json.loads(l)["sample_uid"].rsplit(":", 1)[1]): json.loads(l) for l in open(p)} if os.path.exists(p) else {}


def units(arms, budget, lines_range):
    u32 = measure(DATA, TOK, 100, 32, TASKS, 64, 32)
    recs = []
    for t in TASKS:
        lines = [json.loads(l) for l in open(os.path.join(DATA, t, "validation.jsonl"))]
        rows = {a: load(path(a, budget), t) for a in arms}
        for i in lines_range:
            if any(i not in rows[a] for a in arms):
                continue
            for k, ref in enumerate(lines[i]["outputs"]):
                recs.append(dict(task=t, i=i, cls=u32["tasks"][t]["samples"][i]["cls"][k],
                                 ok={a: int(ref.lower() in postprocess_pred(rows[a][i]["pred"]).lower()) for a in arms}))
    return recs


def booter(recs):
    samples = sorted({(r["task"], r["i"]) for r in recs})
    by = {s: [r for r in recs if (r["task"], r["i"]) == s] for s in samples}
    rng = np.random.default_rng(SEED)
    picks = []
    for _ in range(B_BOOT):
        p = []
        for t in TASKS:
            ss = [s for s in samples if s[0] == t]
            p += [ss[j] for j in rng.integers(0, len(ss), len(ss))] if ss else []
        picks.append([r for s in p for r in by[s]])

    def ci(stat, level=95):
        v = [stat(u) for u in picks]
        v = [x for x in v if np.isfinite(x)]
        q = (100 - level) / 2
        return tuple(np.percentile(v, [q, 100 - q]))
    return ci, len(samples)


def acc(u, a):
    return np.mean([r["ok"][a] for r in u]) if u else np.nan


def G(u, a):
    return acc([r for r in u if r["cls"] == "in0"], a) - acc([r for r in u if r["cls"] == "repair"], a)


def main():
    out = []

    def say(s):
        print(s)
        out.append(s)
    # determinism check on lines 0..19 against round 1 (same samples, same arm)
    for a, rel in R1_DIRS.items():
        same = tot = 0
        for t in TASKS:
            r1, r2 = load(os.path.join(ROUND1, rel), t), load(path(a, 512), t)
            for i in range(20):
                if i in r1 and i in r2:
                    same += r1[i]["pred"] == r2[i]["pred"]
                    tot += 1
        say(f"determinism (lines 0..19 vs round 1) {a:12s}: pred equal {same}/{tot}")
    arms = tuple(ARMS)
    recs = units(arms, 512, range(20, 100))
    ci, ns = booter(recs)
    say(f"rep2 @512: {len(recs)} units, {ns} samples (lines 20..99)")
    for a in arms:
        say(f"  {a:12s} {acc(recs, a):.3f}  in0 {acc([r for r in recs if r['cls'] == 'in0'], a):.3f}  "
            f"repair {acc([r for r in recs if r['cls'] == 'repair'], a):.3f}")

    def d(a, b):
        return lambda u: acc(u, a) - acc(u, b)
    r1 = acc(recs, "cobs16") - acc(recs, "cobs+g2f")
    lo, hi = ci(d("cobs16", "cobs+g2f"))
    v1 = "复现" if lo > 0 else ("方向一致" if r1 > 0 else "未复现")
    say(f"R1 cobs16 - cobs+g2f = {r1:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {v1}")
    r2 = acc(recs, "quest32") - acc(recs, "mean+g2f")
    lo, hi = ci(d("quest32", "mean+g2f"))
    v2 = "复现" if lo >= 0.15 else ("部分" if r2 >= 0.15 else "未复现")
    say(f"R2 quest32 - mean+g2f = {r2:+.3f} [{lo:+.3f}, {hi:+.3f}] (rule on the LOWER bound >= 0.15) -> {v2}")
    gm, gc, gq = G(recs, "mean"), G(recs, "cobs"), G(recs, "quest32")
    l1, h1 = ci(lambda u: G(u, "mean") - G(u, "cobs"))
    l2, h2 = ci(lambda u: G(u, "cobs") - G(u, "quest32"))
    n3 = int(l1 > 0) + int(l2 > 0)   # amendment 2026-10-18 (Codex audit): numpy.bool + numpy.bool is a logical OR
    say(f"R3 G_mean {gm:+.3f}  G_cobs {gc:+.3f}  G_quest {gq:+.3f};  G_mean - G_cobs {gm - gc:+.3f} [{l1:+.3f}, {h1:+.3f}]; "
        f"G_cobs - G_quest {gc - gq:+.3f} [{l2:+.3f}, {h2:+.3f}] -> {['未复现', '部分', '完全复现'][n3]}")
    r4 = acc(recs, "quest32+g2f") - acc(recs, "quest16")
    l90, h90 = ci(d("quest32+g2f", "quest16"), 90)
    l95, h95 = ci(d("quest32+g2f", "quest16"))
    say(f"R4 quest32+g2f - quest16 = {r4:+.3f}; 90% [{l90:+.3f}, {h90:+.3f}] -> "
        f"{'等价' if l90 >= -0.06 and h90 <= 0.06 else '未判定等价'}; 95% [{l95:+.3f}, {h95:+.3f}]")
    for a, b in (("cobs+g2f", "cobs"), ("mean+g2f", "mean")):
        lo, hi = ci(d(a, b))
        say(f"R5 {a} - {b} = {acc(recs, a) - acc(recs, b):+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'复现' if lo > 0 else '未复现'}")
    data = {}
    for s, base, tw in (("cobs", "cobs", "cobs+g2f"), ("mean", "mean", "mean+g2f"), ("quest", "quest32", "quest32+g2f")):
        for t in TASKS:
            rs = sorted({r["i"] for r in recs if r["task"] == t})
            cls = [[r["cls"] for r in recs if r["task"] == t and r["i"] == i] for i in rs]
            b = [[r["ok"][base] for r in recs if r["task"] == t and r["i"] == i] for i in rs]
            c = [[r["ok"][tw] for r in recs if r["task"] == t and r["i"] == i] for i in rs]
            data[s, t] = (cls, b, c)

    def kappa_rule(lo, hi):
        return "相符" if lo <= 1 <= hi and not (lo <= 0.5 <= hi) else ("不成立" if not lo <= 1 <= hi else "不确定")
    _, point, per, kci = analyse(data, boot=B_BOOT, seed=SEED)
    say(f"R6 pooled kappa {point['kappa']:+.3f} [{kci['kappa'][0]:+.3f}, {kci['kappa'][1]:+.3f}] -> "
        f"{kappa_rule(*kci['kappa'])}")
    for s in ("cobs", "mean", "quest"):
        sub = {k: v for k, v in data.items() if k[0] == s}
        _, p_s, _, c_s = analyse(sub, boot=B_BOOT, seed=SEED)
        say(f"R6 {s:5s} kappa {p_s['kappa']:+.3f} [{c_s['kappa'][0]:+.3f}, {c_s['kappa'][1]:+.3f}] -> "
            f"{kappa_rule(*c_s['kappa'])}  (R_repair {p_s['R_repair']:+.2f}, inside {p_s['inside']:+.3f})")
    if all(os.path.exists(path(a, 1024)) for a in R7_ARMS):
        r7 = units(R7_ARMS, 1024, range(20, 60))
        a1 = acc(r7, "cobs16") - acc(r7, "cobs+g2f")
        a2 = acc(r7, "quest32") - acc(r7, "mean+g2f")
        say(f"R7 @1024 ({len(r7)} units, lines 20..59): R1 point {a1:+.3f} (512: {r1:+.3f}) -> "
            f"{'方向一致' if np.sign(a1) == np.sign(r1) else '方向不一致'}; R2 point {a2:+.3f} (512: {r2:+.3f}) -> "
            f"{'方向一致' if np.sign(a2) == np.sign(r2) else '方向不一致'}")
    else:
        say("R7: budget-1024 outputs not present (not run yet)")


if __name__ == "__main__":
    main()
