"""P-gain: the registered descriptive test of the gain law (plan v1.4 §3.4; docs/THEORY_ZH.md §4; reviewer U1, batch-2
corrections). Run only after cluster/aggregate.py has passed its gate for every pair used (its --json outputs are
inputs here and bind the data).

Units come from the frozen unit-length file (cluster/unit_lengths.py v2): one unit per answer reference, class
in0 / repair / out / fixed at the first decode call. A unit is answered iff its reference appears in the post-processed
prediction (case-insensitive substring: RULER's string_match_all credit). For a scorer s (baseline arm s, treatment
s+g2f) and a task t with units, on the evaluated lines:
  pi      = share of repair units;   a_c / a'_c = baseline / treatment accuracy over units of class c;
  G       = a_in0 - a_repair (single-grid straddle penalty, baseline arm only);
  Delta   = mean over units of (treatment - baseline)                       (= the official score difference / 100);
  pred    = sum_f n_repair(f) * G(other fold) / n_units       — G cross-fitted on line-parity folds (removes the shared
            noise between G and Delta, which both contain a_repair);
  kappa   = sum_cells Delta / sum_cells pred   (pooled calibration ratio; the law predicts kappa ~ 1);
  slope   = least squares through the origin of Delta on pred over cells (auxiliary: attenuated by noise in G);
  R_c     = sum_cells (a'_c - a_c) / sum_cells (a_in0 - a_c), c in {repair, out} (the law: R_repair ~ 1, R_out ~ 0);
  inside  = sum_cells (a'_in0 - a_in0) / cells (the law: ~ 0).
P-asym (descriptive; unit_lengths v3 rv/vlen): per scorer, baseline failures of repair units whose grid-0 cut lies
inside the value vs before it (counts pooled over tasks).
Intervals: 2000 stratified bootstrap resamples of samples within each task (units of a sample stay together; the
fold of a sample is its line parity and travels with it; G is re-cross-fitted inside every resample). Descriptive only.
Usage: gain_law.py --out-root DIR --model M --length L --budget B --n N --run-tag TAG --units FILE
       --agg AGG.json [--agg AGG.json ...] [--json OUT]"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aggregate import PLAN, postprocess_pred  # noqa: E402

SCORERS = ("cobs", "nf_pt1", "quest32")


def cell_stats(cls, b, c, idx, fold):
    """cls/b/c: per sample lists (per unit); idx: sample indices (with repeats); fold: per sample 0/1.
    Returns dict with pi, G, Delta, pred, and per-class sums for R."""
    acc = {}
    for k in ("in0", "repair", "out"):
        for f in (0, 1, None):
            sel = [(bb, cc) for i in idx if f is None or fold[i] == f
                   for cl, bb, cc in zip(cls[i], b[i], c[i]) if cl == k]
            acc[k, f] = (len(sel), sum(x for x, _ in sel), sum(y for _, y in sel))
    n_units = sum(len(cls[i]) for i in idx)
    delta = sum(cc - bb for i in idx for bb, cc in zip(b[i], c[i])) / n_units

    def G(f):
        (ni, bi, _), (nr, br, _) = acc["in0", f], acc["repair", f]
        return bi / ni - br / nr if ni and nr else 0.0
    pred = sum(acc["repair", f][0] * G(1 - f) for f in (0, 1)) / n_units
    out = dict(pi=acc["repair", None][0] / n_units, G=G(None), delta=delta, pred=pred, n_units=n_units)
    for k in ("in0", "repair", "out"):
        n, sb, sc = acc[k, None]
        out[k] = (n, sb / n if n else None, sc / n if n else None)
    return out


def summarise(cells):
    D = sum(x["delta"] for x in cells)
    P = sum(x["pred"] for x in cells)
    num = sum(x["delta"] * x["pred"] for x in cells)
    den = sum(x["pred"] ** 2 for x in cells)
    res = dict(kappa=D / P if P else float("nan"), slope=num / den if den else float("nan"))
    for k in ("repair", "out"):
        u = [(x[k][2] - x[k][1], x["in0"][1] - x[k][1]) for x in cells if x[k][0] and x["in0"][0]]
        res[f"R_{k}"] = sum(a for a, _ in u) / sum(g for _, g in u) if u and sum(g for _, g in u) else float("nan")
    ins = [x["in0"][2] - x["in0"][1] for x in cells if x["in0"][0]]
    res["inside"] = sum(ins) / len(ins) if ins else float("nan")
    return res


def analyse(data, boot=2000, seed=0):
    """data: {(scorer, task): (cls, b, c)} with the same sample count per task across scorers."""
    fold_of = {}
    full = {}
    for key, (cls, b, c) in data.items():
        n = len(cls)
        fold_of[key] = [i % 2 for i in range(n)]
        full[key] = cell_stats(cls, b, c, list(range(n)), fold_of[key])
    point = summarise(list(full.values()))
    per_scorer = {s: summarise([v for (ss, _), v in full.items() if ss == s]) for s in {k[0] for k in full}}
    rng = np.random.default_rng(seed)
    tasks = sorted({k[1] for k in data})
    draws = []
    for _ in range(boot):
        idx = {t: rng.integers(0, len(next(v for k, v in data.items() if k[1] == t)[0]), size=len(
            next(v for k, v in data.items() if k[1] == t)[0])).tolist() for t in tasks}
        cells = [cell_stats(cls, b, c, idx[key[1]], fold_of[key]) for key, (cls, b, c) in data.items()]
        draws.append(summarise(cells))
    ci = {k: [float(np.nanpercentile([d[k] for d in draws], q)) for q in (2.5, 97.5)] for k in point}
    return full, point, per_scorer, ci


def load(a, units, scorer, task, n):
    arm_b, arm_t = PLAN["arms"][scorer], PLAN["arms"][scorer + "+g2f"]
    rows = {}
    for name, arm in (("b", arm_b), ("t", arm_t)):
        d = os.path.join(a.out_root, "pred", a.model, str(a.length), f"budget{a.budget}", f"b{arm['block']}", a.run_tag)
        rows[name] = {json.loads(l)["sample_uid"]: json.loads(l) for l in
                      open(os.path.join(d, f"{arm['method']}@{a.budget}", f"{task}.jsonl"))}
    data_dir = os.path.join(a.out_root, "data", a.model, str(a.length))
    sha = units["tasks"][task]["data_sha256"]
    lines = [json.loads(l) for l in open(os.path.join(data_dir, task, "validation.jsonl"))][:n]
    cls, b, c, asym = [], [], [], []
    for i, ln in enumerate(lines):
        uid = f"{sha[:16]}:{task}:{i}"
        smp = units["tasks"][task]["samples"][i]
        if len(smp["cls"]) != len(ln["outputs"]):
            raise SystemExit(f"{task} line {i}: unit count differs from references")
        cls.append(smp["cls"])
        asym.append([None if cl != "repair" else ("inside" if rv < vl else "before")
                     for cl, rv, vl in zip(smp["cls"], smp["rv"], smp["vlen"])])
        for name, out in (("b", b), ("t", c)):
            pred = postprocess_pred(rows[name][uid]["pred"]).lower()
            out.append([int(r.lower() in pred) for r in ln["outputs"]])
    return cls, b, c, asym


def asym_table(data, asym):
    """P-asym (descriptive): baseline failures of repair units, cut inside the value vs before it, per scorer."""
    out = {}
    for (s, t), (cls, b, c) in data.items():
        for flags, bb in zip(asym[s, t], b):
            for f, x in zip(flags, bb):
                if f:
                    k = out.setdefault(s, {"inside": [0, 0], "before": [0, 0]})[f]
                    k[0] += 1 - x
                    k[1] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("out-root", "model", "run-tag", "units"):
        ap.add_argument(f"--{k}", required=True)
    for k in ("length", "budget", "n"):
        ap.add_argument(f"--{k}", type=int, required=True)
    ap.add_argument("--agg", action="append", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    units = json.load(open(a.units))
    usha = hashlib.sha256(open(a.units, "rb").read()).hexdigest()
    passed = {}
    for p in a.agg:
        r = json.load(open(p))
        if (r.get("model"), r.get("length"), r.get("budget"), r.get("n")) != (a.model, a.length, a.budget, a.n):
            raise SystemExit(f"REFUSED: {p} is another cell")
        if any(units["tasks"][t]["data_sha256"] != h for t, h in r["data_sha256"].items()):
            raise SystemExit(f"REFUSED: {p} was computed on other data than the unit file")
        passed[tuple(r["pair"][:2])] = r
    data, asym = {}, {}
    for s in SCORERS:
        if (s + "+g2f", s) not in passed:
            print(f"# {s}: no gated aggregate result for {s}+g2f vs {s}; skipped")
            continue
        for t in PLAN["tasks"]:
            if any(smp["cls"] for smp in units["tasks"][t]["samples"][:a.n]):
                *data[s, t], asym[s, t] = load(a, units, s, t, a.n)
    if not data:
        raise SystemExit("REFUSED: no scorer with a gated pair")
    full, point, per_scorer, ci = analyse(data)
    print(f"P-gain (descriptive; unit file {usha[:16]}): cell {a.model} L={a.length} budget={a.budget} n={a.n}")
    print(f"  {'scorer':8s} {'task':16s} {'pi':>6s} {'G':>7s} {'R_rep':>7s} {'Delta':>7s} {'pred':>7s}")
    for (s, t), x in sorted(full.items()):
        rr = (x["repair"][2] - x["repair"][1]) / x["G"] if x["repair"][0] and x["G"] > 0.05 else float("nan")
        print(f"  {s:8s} {t:16s} {x['pi']:6.3f} {x['G']:+7.3f} {rr:7.2f} {x['delta']:+7.3f} {x['pred']:+7.3f}")
    for k in ("kappa", "slope", "R_repair", "R_out", "inside"):
        print(f"  pooled {k:8s} {point[k]:+.3f}  95% [{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]")
    for s, v in sorted(per_scorer.items()):
        print(f"  {s:8s} kappa {v['kappa']:+.3f}  R_repair {v['R_repair']:+.3f}  R_out {v['R_out']:+.3f}")
    at = asym_table(data, asym)
    for s, v in sorted(at.items()):
        fi, ni = v["inside"]
        fb, nb = v["before"]
        print(f"  P-asym {s:8s} baseline failures, cut inside value {fi}/{ni}"
              f"{f' ({fi / ni:.3f})' if ni else ''} vs before value {fb}/{nb}{f' ({fb / nb:.3f})' if nb else ''}")
    if a.json:
        json.dump(dict(units_sha256=usha, point=point, ci95=ci, per_scorer=per_scorer, p_asym=asym_table(data, asym),
                       cells={f"{s}|{t}": x for (s, t), x in full.items()}), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
