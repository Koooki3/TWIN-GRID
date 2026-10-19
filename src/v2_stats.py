"""Statistics for plan v2 (docs/PLAN_V2_COGNITIVE_ZH.md r2, §3.3–3.4; Codex reviews V1–V7, R1-a, R1-b). Pure functions
on answer-unit records; no model, no I/O. A record is dict(task, i, cls, ok={arm: 0/1}); cls in {in0, repair, out,
fixed} from the frozen 32-grid unit file. Samples (task, i) are the exchange / resampling unit; all answer units of a
sample and all arms move together.

H1/H2  diff(a, b)      = mean over all units of ok[a] - ok[b]                     (answer-unit weighted)
H3     gdiff(a, b)     = G_a - G_b, G_x = acc_x(in0) - acc_x(repair)               (fixed / out excluded)
Permutation (sharp exchangeability null: within each sample the joint outcome law is invariant to swapping the two arm
labels): each sample's two outcome vectors are swapped with probability 1/2. Both statistics are linear in per-sample
contributions under a swap (class counts are unchanged by swapping), so T* = sum_k s_k d_k with s_k = +/-1.
One-sided p = (#{T* >= T} + 1) / (N + 1), ties extreme. N = 100000, seed 20261019 + k.
Bootstrap: task-stratified, whole-sample, B = 4000, seed 20261019; a draw whose statistic is undefined is INVALID (never
dropped-and-renormalised, never redrawn); status() propagates: invalid fraction f <= 1% -> percentile CI on the valid
draws ("conditional on valid draws", exploratory), f > 1% -> "不可判定".
H4 (exploratory): cells C fixed from the unit file (calibration in0 >= 20 and repair >= 20 per task in S_unit);
pred_law = pi_main * G_calib (R fixed to 1); baselines B1 (pooled calibration Delta over C, unit weighted) and B2
(per-cell calibration Delta, transfer reference); E_X = sum w |Delta_main - pred_X| / sum w; geometry (pi_main, w)
re-computed in every draw; main and calibration resampled independently. kappa = sum_c Delta_c / sum_c pred_c (r2).
"""
import numpy as np

SEED = 20261019
N_PERM = 100000
B_BOOT = 4000
MAX_INVALID = 0.01
KAPPA_MIN = 0.005
TOL_3PP = 0.03
MIN_CLASS_H3 = 50
MIN_CLASS_CALIB = 20


def by_sample(recs):
    out = {}
    for r in recs:
        out.setdefault((r["task"], r["i"]), []).append(r)
    return out


def acc(u, a):
    return sum(r["ok"][a] for r in u) / len(u) if u else None


def diff(u, a, b):
    return (sum(r["ok"][a] - r["ok"][b] for r in u) / len(u)) if u else None


def G(u, a):
    i0 = [r for r in u if r["cls"] == "in0"]
    rp = [r for r in u if r["cls"] == "repair"]
    if not i0 or not rp:
        return None
    return acc(i0, a) - acc(rp, a)


def gdiff(u, a, b):
    ga, gb = G(u, a), G(u, b)
    return None if ga is None or gb is None else ga - gb


# ---------------------------------------------------------------- permutation (H1–H3)
def contributions(recs, a, b, kind):
    """Per-sample d_k with T = sum_k d_k for the observed labelling; kind 'diff' or 'gdiff'."""
    S = by_sample(recs)
    if kind == "diff":
        n = len(recs)
        return np.array([sum(r["ok"][a] - r["ok"][b] for r in u) / n for u in S.values()], float)
    n0 = sum(r["cls"] == "in0" for r in recs)
    nr = sum(r["cls"] == "repair" for r in recs)
    if n0 == 0 or nr == 0:
        return None
    return np.array([sum(r["ok"][a] - r["ok"][b] for r in u if r["cls"] == "in0") / n0
                     - sum(r["ok"][a] - r["ok"][b] for r in u if r["cls"] == "repair") / nr for u in S.values()], float)


def perm_p(d, k, n_perm=N_PERM, chunk=5000):
    """One-sided (greater) Monte Carlo p for T = sum d under sample-level sign flips; seed SEED + k."""
    t = float(d.sum())
    rng = np.random.default_rng(SEED + k)
    ge, done = 0, 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        s = rng.integers(0, 2, size=(m, len(d)), dtype=np.int8) * 2 - 1
        ge += int(((s @ d) >= t - 1e-12).sum())
        done += m
    return (ge + 1) / (n_perm + 1), t


def holm(ps, alpha=0.05):
    """ps: dict name -> p or None (missing / not estimable -> p = 1, family size unchanged)."""
    names = list(ps)
    p = {n: (1.0 if ps[n] is None else ps[n]) for n in names}
    order = sorted(names, key=lambda n: p[n])
    m, run, adj = len(names), 0.0, {}
    for j, n in enumerate(order):
        run = max(run, min(1.0, (m - j) * p[n]))
        adj[n] = run
    return {n: dict(p=ps[n], p_holm=adj[n], reject=adj[n] <= alpha,
                    state=("未判定" if ps[n] is None else ("拒绝" if adj[n] <= alpha else "未拒绝"))) for n in names}


# ---------------------------------------------------------------- bootstrap and invalid-draw propagation
def status(values, level=95, max_invalid=MAX_INVALID):
    """values: list of floats or None (invalid draw). Returns dict(state, lo, hi, f, n_valid)."""
    n = len(values)
    v = [x for x in values if x is not None]
    f = (n - len(v)) / n if n else 1.0
    if not v or f > max_invalid + 1e-12:
        return dict(state="不可判定", lo=None, hi=None, f=f, n_valid=len(v))
    q = (100 - level) / 2
    lo, hi = np.percentile(v, [q, 100 - q])
    return dict(state=("有效" if f == 0 else "有效（以抽取有效为条件，探索性）"), lo=float(lo), hi=float(hi), f=f,
                n_valid=len(v))


def draws(recs, rng, n_boot):
    """Yield task-stratified whole-sample bootstrap resamples (lists of records)."""
    S = by_sample(recs)
    tasks = sorted({t for t, _ in S})
    per = {t: [s for s in S if s[0] == t] for t in tasks}
    for _ in range(n_boot):
        yield [r for t in tasks for j in rng.integers(0, len(per[t]), len(per[t])) for r in S[per[t][j]]]


def boot_stat(recs, stat, n_boot=B_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    return [stat(u) for u in draws(recs, rng, n_boot)]


# ---------------------------------------------------------------- H4
def cells(calib_recs, s_unit, min_class=MIN_CLASS_CALIB):
    """C from the unit classes only (outcome independent)."""
    C, excluded = [], {}
    for t in s_unit:
        u = [r for r in calib_recs if r["task"] == t]
        n0 = sum(r["cls"] == "in0" for r in u)
        nr = sum(r["cls"] == "repair" for r in u)
        if n0 >= min_class and nr >= min_class:
            C.append(t)
        else:
            excluded[t] = f"calibration in0 {n0}, repair {nr} (need >= {min_class} each)"
    return C, excluded


def missing_classes(calib, C):
    """Per C cell, which of in0 / repair is absent in this calibration (re)sample."""
    out = {}
    for t in C:
        cl = {r["cls"] for r in calib if r["task"] == t}
        miss = [c for c in ("in0", "repair") if c not in cl]
        if miss:
            out[t] = miss
    return out


def h4_point(main, calib, C, base, twin):
    """Point quantities on one (main, calib) pair; None if a C cell lacks in0 or repair in calib (invalid draw)."""
    if not C:
        return None
    cell = {}
    for t in C:
        um = [r for r in main if r["task"] == t]
        uc = [r for r in calib if r["task"] == t]
        g = G(uc, base)
        if g is None or not um:
            return None
        pi = sum(r["cls"] == "repair" for r in um) / len(um)
        cell[t] = dict(w=len(um), pi=pi, d_main=diff(um, twin, base), g_calib=g, d_calib=diff(uc, twin, base),
                       pred_law=pi * g,
                       decomp={c: (sum(r["cls"] == c for r in um) / len(um)) *
                               ((diff([r for r in um if r["cls"] == c], twin, base)) or 0.0)
                               for c in ("in0", "repair", "out", "fixed")})
    uc_all = [r for r in calib if r["task"] in C]
    b1 = diff(uc_all, twin, base)
    W = sum(c["w"] for c in cell.values())

    def E(key):
        pred = {"B0": lambda c: 0.0, "B1": lambda c: b1}.get(key, lambda c: c[key])
        return sum(c["w"] * abs(c["d_main"] - pred(c)) for c in cell.values()) / W
    e_law, e_b0, e_b1, e_b2 = E("pred_law"), E("B0"), E("B1"), E("d_calib")
    # registered r2 kappa (Codex I4): unweighted sums over the cells of C, threshold on |sum pred|
    sum_pred = sum(c["pred_law"] for c in cell.values())
    sum_delta = sum(c["d_main"] for c in cell.values())
    return dict(cells=cell, b1=b1, e_law=e_law, e_b0=e_b0, e_b1=e_b1, e_b2=e_b2, rel=e_law - e_b1,
                rel_b2=e_law - e_b2, rel_b0=e_law - e_b0, sum_pred=sum_pred, sum_delta=sum_delta)


def kappa_of(q, sign_ref=None):
    """kappa = sum_c Delta_c / sum_c pred_c (plan r2); not estimable if |sum pred| < 0.005; in a bootstrap draw also
    invalid if sum pred has the opposite sign to the original sample's (and, by the caller, if the draw is invalid)."""
    if q is None or abs(q["sum_pred"]) < KAPPA_MIN:
        return None
    if sign_ref is not None and np.sign(q["sum_pred"]) != np.sign(sign_ref):
        return None
    return q["sum_delta"] / q["sum_pred"]


def abs_label(st):
    if st["state"] == "不可判定":
        return "不可判定"
    if st["hi"] <= TOL_3PP:
        return "在 C 上，答案单元加权的平均绝对预测误差的 95% 上界不超过 3pp（不保证每格）"
    if st["lo"] > TOL_3PP:
        return "加权平均绝对误差明确超过 3pp"
    return "未证实达到预定的加权平均 3pp 精度"


def rel_label(st, n_cells):
    if n_cells < 2:
        return "不适用（|C| < 2，B1 与 B2 重合）"
    if st["state"] == "不可判定":
        return "不可判定"
    if st["hi"] < 0:
        return "在 C 上，结构律的加权平均绝对误差小于常数预测"
    if st["lo"] > 0:
        return "大于常数预测"
    return "不确定"


READOUTS = ("e_law", "e_b0", "e_b1", "e_b2", "rel", "rel_b0", "rel_b2")


def h4(main, calib, s_unit, base="cobs", twin="cobs+g2f", n_boot=B_BOOT, seed=SEED):
    """H4 readouts. Output keys: C, excluded, coverage (always); for |C| >= 1 also point, kappa_point, ci (every
    readout of READOUTS and kappa: state / lo / hi / f / n_valid), invalid_by_cell (per C cell: draws missing in0 /
    repair), abs_label, rel_label, state. E_B0 (zero prediction) is reported as a reference, not a decision readout."""
    C, excluded = cells(calib, s_unit)
    n_su = sum(1 for r in main if r["task"] in s_unit)
    n_c = sum(1 for r in main if r["task"] in C)
    out = dict(C=C, excluded=excluded, coverage=dict(cells=len(C), s_unit_tasks=len(s_unit), main_units_in_C=n_c,
                                                     main_units_in_S_unit=n_su,
                                                     unit_share=(n_c / n_su if n_su else None)))
    if not C:
        out["state"] = "不可估计（|C| = 0）"
        return out
    pt = h4_point(main, calib, C, base, twin)
    out["point"] = pt
    kp = kappa_of(pt)
    out["kappa_point"] = kp
    rng = np.random.default_rng(seed)
    gm, gc = draws(main, rng, n_boot), draws(calib, rng, n_boot)
    vals = {k: [] for k in READOUTS + ("kappa",)}
    inv = {t: {"in0": 0, "repair": 0} for t in C}
    for um, uc in zip(gm, gc):
        for t, miss in missing_classes(uc, C).items():
            for c in miss:
                inv[t][c] += 1
        q = h4_point(um, uc, C, base, twin)
        for k in READOUTS:
            vals[k].append(None if q is None else q[k])
        # kappa: union of the base invalid rule and the kappa-specific rules
        vals["kappa"].append(None if kp is None else kappa_of(q, pt["sum_pred"]))
    out["invalid_by_cell"] = inv
    st = {k: status(v) for k, v in vals.items()}
    if kp is None:
        st["kappa"] = dict(state="不可估计", lo=None, hi=None, f=None, n_valid=0)
    out.update(ci=st, abs_label=abs_label(st["e_law"]), rel_label=rel_label(st["rel"], len(C)),
               state=("单格" if len(C) == 1 else "完整"))
    return out


def h3_estimable(main_recs, min_class=MIN_CLASS_H3):
    n0 = sum(r["cls"] == "in0" for r in main_recs)
    nr = sum(r["cls"] == "repair" for r in main_recs)
    return n0 >= min_class and nr >= min_class, n0, nr
