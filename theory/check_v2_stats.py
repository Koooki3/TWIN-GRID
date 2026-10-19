"""Isolated synthetic tests of src/v2_stats.py (plan v2 r2 §4; Codex r2 review boundary list). No model, no data files.
Covers: known / zero effect; linear permutation identity for H1 and H3 against direct recomputation; Holm with a missing
member; invalid-draw propagation at 40/4000 vs 41/4000; |C| = 0 and |C| = 1; kappa (registered sum form: oracle .75, |sum pred| 0.0049 / 0.005 / .006),
sign flip and union with the base invalid rule; the three 3pp states; G <= 0 predictions; H3 interval invalidity does
not change the permutation p; structural law beats the constant baseline when it holds exactly.
Exit 1 on any failure."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import v2_stats as V  # noqa: E402

bad = []


def check(cond, what):
    print(("OK   " if cond else "FAIL ") + what)
    if not cond:
        bad.append(what)


def make(rng, tasks, n, units, p, cls_p=(0.6, 0.35, 0.05)):
    """p: dict arm -> dict cls -> success prob."""
    recs = []
    for t in tasks:
        for i in range(n):
            for _ in range(units):
                c = rng.choice(["in0", "repair", "out"], p=list(cls_p))
                recs.append(dict(task=t, i=i, cls=c, ok={a: int(rng.random() < p[a][c]) for a in p}))
    return recs


rng = np.random.default_rng(1)
T = [f"t{k}" for k in range(5)]
# 1. known effect / zero effect
eff = make(rng, T, 60, 2, {"a": dict(in0=.8, repair=.8, out=.8), "b": dict(in0=.6, repair=.6, out=.6)})
d = V.contributions(eff, "a", "b", "diff")
p, t = V.perm_p(d, 1, n_perm=20000)
check(abs(t - V.diff(eff, "a", "b")) < 1e-12, "diff contributions sum to the observed statistic")
check(p < 0.001, f"known effect: p = {p:.5f} < 0.001")
same = [dict(r, ok={"a": r["ok"]["a"], "b": r["ok"]["a"]}) for r in eff]
p0, _ = V.perm_p(V.contributions(same, "a", "b", "diff"), 1, n_perm=2000)
check(p0 == 1.0, "identical arms: p = 1 (ties extreme)")
null = make(rng, T, 60, 2, {"a": dict(in0=.7, repair=.5, out=.5), "b": dict(in0=.7, repair=.5, out=.5)})
pn, _ = V.perm_p(V.contributions(null, "a", "b", "diff"), 2, n_perm=20000)
check(pn > 0.01, f"exchangeable null: p = {pn:.3f} not tiny")

# 2. linear permutation identity (H1 and H3) against direct recomputation under explicit swaps
S = V.by_sample(eff)
keys = list(S)
for kind, fn in (("diff", V.diff), ("gdiff", V.gdiff)):
    dk = V.contributions(eff, "a", "b", kind)
    ok = True
    for trial in range(20):
        s = rng.integers(0, 2, len(keys)) * 2 - 1
        sw = []
        for sk, k in zip(s, keys):
            for r in S[k]:
                sw.append(r if sk > 0 else dict(r, ok={"a": r["ok"]["b"], "b": r["ok"]["a"]}))
        ok &= abs(float(s @ dk) - fn(sw, "a", "b")) < 1e-12
    check(ok, f"{kind}: sign-flip identity equals direct recomputation (20 random swaps)")

# 3. Holm
h = V.holm({"H1": 0.001, "H2": 0.04, "H3": None})
check(h["H3"]["state"] == "未判定" and h["H3"]["p_holm"] == 1.0, "Holm: missing member -> 未判定, p = 1")
check(abs(h["H1"]["p_holm"] - 0.003) < 1e-12 and h["H1"]["reject"], "Holm: smallest p x 3")
check(abs(h["H2"]["p_holm"] - 0.08) < 1e-12 and not h["H2"]["reject"], "Holm: second p x 2, family size kept at 3")

# 4. invalid-draw propagation 40/4000 vs 41/4000
v40 = [0.1] * 3960 + [None] * 40
v41 = [0.1] * 3959 + [None] * 41
s40, s41 = V.status(v40), V.status(v41)
check(s40["state"].startswith("有效（以抽取有效为条件") and abs(s40["f"] - 0.01) < 1e-12, "40/4000 invalid -> conditional CI")
check(s41["state"] == "不可判定" and s41["lo"] is None, "41/4000 invalid -> 不可判定")
check(V.status([0.1] * 4000)["state"] == "有效", "0 invalid -> 有效")

# 5. H4: |C| = 0 and |C| = 1, structural law, G <= 0
law = {"cobs": dict(in0=.8, repair=.4, out=.3), "cobs+g2f": dict(in0=.8, repair=.8, out=.3)}
main = make(rng, T, 100, 2, law)
calib = make(rng, T, 100, 2, law)
few = make(rng, T, 5, 1, law)
r0 = V.h4(main, few, T, n_boot=200)
check(r0["C"] == [] and r0["state"].startswith("不可估计"), "|C| = 0 -> 不可估计")
calib1 = [r for r in calib if r["task"] == "t0"] + [r for r in few if r["task"] != "t0"]
r1 = V.h4(main, calib1, T, n_boot=200)
check(r1["C"] == ["t0"] and r1["rel_label"].startswith("不适用") and r1["state"] == "单格", "|C| = 1 -> relative label 不适用")
# structural law holding exactly (R = 1, out term small) with varying pi across tasks
mainv, calibv = [], []
for k, t in enumerate(T):
    cp = (0.85 - 0.12 * k, 0.15 + 0.12 * k - 0.01, 0.01)
    mainv += make(rng, [t], 150, 2, law, cp)
    calibv += make(rng, [t], 150, 2, law, cp)
r = V.h4(mainv, calibv, T, n_boot=300)
check(r["point"]["e_law"] < r["point"]["e_b1"], f"law holds: E_law {r['point']['e_law']:.4f} < E_B1 {r['point']['e_b1']:.4f}")
neg = {"cobs": dict(in0=.4, repair=.6, out=.3), "cobs+g2f": dict(in0=.4, repair=.6, out=.3)}
rn = V.h4(make(rng, T, 60, 2, neg), make(rng, T, 60, 2, neg), T, n_boot=100)
check(all(c["g_calib"] < 0 and c["pred_law"] < 0 for c in rn["point"]["cells"].values()), "G <= 0 still predicted (negative pred)")
dec = r["point"]["cells"]["t0"]
check(abs(sum(dec["decomp"].values()) - dec["d_main"]) < 1e-12, "class decomposition sums to Delta_main")

# 6. kappa (registered r2 form: sum_c Delta_c / sum_c pred_c, threshold on |sum pred|; Codex I4)
check(V.kappa_of(dict(sum_pred=0.0049, sum_delta=0.01)) is None, "kappa: |sum pred| 0.0049 -> 不可估计")
check(V.kappa_of(dict(sum_pred=0.005, sum_delta=0.01)) is not None, "kappa: |sum pred| exactly 0.005 -> estimable")
check(V.kappa_of(dict(sum_pred=0.006, sum_delta=0.01)) is not None, "kappa: two cells with pred .003 (sum .006) -> estimable")
check(V.kappa_of(dict(sum_pred=-0.02, sum_delta=0.01), sign_ref=0.02) is None, "kappa: sign flip vs point -> invalid draw")
check(V.kappa_of(None, 0.02) is None, "kappa: base-invalid draw is also kappa-invalid (union)")


def oracle_cells():
    """Two cells, main weights 10 / 30, Delta_main .10 / .20, pred .20 / .20 -> registered kappa .75 (weighted .875)."""
    main_, cal_ = [], []
    for t, n, k in (("o0", 10, 1), ("o1", 30, 6)):
        for i in range(n):                         # all repair: pi = 1; base wrong, twin right on k of n
            main_.append(dict(task=t, i=i, cls="repair", ok={"cobs": 0, "cobs+g2f": int(i < k)}))
        for i in range(25):                        # calibration G = acc(in0) - acc(repair) = .6 - .4 = .2
            cal_.append(dict(task=t, i=100 + i, cls="in0", ok={"cobs": int(i < 15), "cobs+g2f": int(i < 15)}))
            cal_.append(dict(task=t, i=200 + i, cls="repair", ok={"cobs": int(i < 10), "cobs+g2f": int(i < 10)}))
    return main_, cal_


om, oc = oracle_cells()
op = V.h4_point(om, oc, ["o0", "o1"], "cobs", "cobs+g2f")
check(abs(op["sum_pred"] - 0.4) < 1e-12 and abs(op["sum_delta"] - 0.3) < 1e-12, "kappa oracle: sums .4 / .3")
check(abs(V.kappa_of(op) - 0.75) < 1e-12, f"kappa oracle: registered kappa .75 (got {V.kappa_of(op):.4f}; weighted would be .875)")
check(abs(op["e_b0"] - (10 * 0.1 + 30 * 0.2) / 40) < 1e-12, "B0: zero-prediction error = weighted mean |Delta|")
ro = V.h4(om, oc, ["o0", "o1", "o2"], n_boot=50)
cov = ro["coverage"]
check(cov["cells"] == 2 and cov["s_unit_tasks"] == 3 and cov["main_units_in_C"] == 40 and abs(cov["unit_share"] - 1.0) < 1e-12,
      "I5: coverage reported (cells, S_unit tasks, unit share)")
check(set(ro["invalid_by_cell"]) == {"o0", "o1"} and all(set(v) == {"in0", "repair"} for v in ro["invalid_by_cell"].values()),
      "I5: per-cell invalid counts (in0 / repair) reported")
check(all(k in ro["ci"] for k in V.READOUTS + ("kappa",)), "I5: every readout (incl. E_B0, E_B1, E_B2) has an interval state")
check("coverage" in r0 and r0["coverage"]["cells"] == 0, "I5: |C| = 0 still reports coverage")

# 7. three 3pp states
check(V.abs_label(dict(state="有效", lo=0.01, hi=0.029)).startswith("在 C 上"), "3pp: upper <= .03 -> bound met (weighted mean)")
check(V.abs_label(dict(state="有效", lo=0.031, hi=0.05)) == "加权平均绝对误差明确超过 3pp", "3pp: lower > .03 -> exceeds")
check(V.abs_label(dict(state="有效", lo=0.02, hi=0.04)).startswith("未证实达到"), "3pp: straddling -> 未证实达到（not a negation）")
check(V.abs_label(dict(state="不可判定")) == "不可判定", "3pp: 不可判定 propagates")

# 8. H3 interval invalidity does not change the permutation p
sparse = make(rng, T, 40, 1, {"a": dict(in0=.8, repair=.3, out=.3), "b": dict(in0=.8, repair=.7, out=.3)},
              cls_p=(0.9, 0.08, 0.02))
# exactly one repair unit in the whole set: a resample misses it with probability (39/40)^40 ~ 0.36 >> 1%
sparse = [dict(r, cls="in0") for r in sparse]
sparse[0] = dict(sparse[0], cls="repair")
dk = V.contributions(sparse, "a", "b", "gdiff")
check(dk is not None, "H3 sparse case: observed in0 and repair both present (statistic defined)")
pa, _ = V.perm_p(dk, 3, n_perm=5000)
bs = V.boot_stat(sparse, lambda u: V.gdiff(u, "a", "b"), n_boot=400)
st = V.status(bs)
check(st["state"] == "不可判定", f"H3 sparse case: bootstrap interval 不可判定 (invalid fraction {st['f']:.3f} > 1%)")
pb, _ = V.perm_p(V.contributions(sparse, "a", "b", "gdiff"), 3, n_perm=5000)
check(pa == pb, f"H3: permutation p ({pa:.4f}) unchanged although the interval is 不可判定")

print("PROBLEMS", bad if bad else "none")
sys.exit(1 if bad else 0)
