"""Phase-2 confirmatory analysis for prereg v3 (single look). Runs the full integrity gate first (which also verifies
that this file, the gate, Newcombe, tasks.py and the prereg text equal the frozen SPEC hashes); refuses on any gate
failure or missing key. Then, on strict scores:
  primary   cobs+g2f vs cobs: exact two-sided McNemar (alpha 0.05), Newcombe method-10 95% interval;
            pre-registered sensitivity: two-sided mid-p (does not change the decision);
  secondary nf_pt1+g2f vs nf_pt1, quest+g2f vs quest: exact McNemar, Holm at 0.05;
  descriptive per task and by grid-0 / grid-1 straddle status;
  secondary ENDPOINT (descriptive only, no test): containment (tasks.score re-computed from out/answers) per arm as
  count/denominator/rate, and improve/worsen counts for the three pre-specified X+g2f vs X pairs.
Usage: phase2_analysis.py SPEC.json PHASE2.jsonl"""
import json
import os
import sys

from scipy.stats import binom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase2_gate  # noqa: E402
from newcombe import newcombe10  # noqa: E402
from tasks import score  # noqa: E402  (containment, the frozen secondary endpoint)


def exact_p(g, m):
    return 1.0 if m == 0 else min(1.0, 2 * binom.cdf(min(g, m - g), m, 0.5))


def midp(g, m):
    if m == 0:
        return 1.0
    k = min(g, m - g)
    return min(1.0, max(0.0, 2 * binom.cdf(k, m, 0.5) - binom.pmf(k, m, 0.5)))   # from the unclipped exact value


def table(rows, a1, a0):
    a = b = c = d = 0
    for r in rows:
        x, y = int(r["res"][a1]["strict"]), int(r["res"][a0]["strict"])
        a += x and y
        b += x and not y
        c += y and not x
        d += not x and not y
    return a, b, c, d


def contrast(rows, a1, a0):
    a, b, c, d = table(rows, a1, a0)
    m = b + c
    D, lo, hi = newcombe10(a, b, c, d)
    return dict(cells=(a, b, c, d), acc1=(a + b) / len(rows), acc0=(a + c) / len(rows), diff=D, ci95=(lo, hi),
                p_exact=exact_p(b, m), p_midp=midp(b, m))


def main(spec_path, path):
    spec = json.load(open(spec_path))
    rows, missing, bad = phase2_gate.check(spec, path, partial=False)
    if bad or missing:
        print("REFUSED: gate failed or keys missing — no confirmatory analysis")
        for x in bad[:20]:
            print("  -", x)
        sys.exit(2)
    B = spec["fixed"]["budget"]
    arm = lambda m: f"{m}@{B}"  # noqa: E731
    out = {}
    out["primary"] = contrast(rows, arm("cobs+g2f"), arm("cobs"))
    out["primary"]["decision"] = "reject H0" if out["primary"]["p_exact"] <= 0.05 else "do not reject H0"
    sec = {k: contrast(rows, arm(f"{k}+g2f"), arm(k)) for k in ("nf_pt1", "quest")}
    order = sorted(sec, key=lambda k: sec[k]["p_exact"])
    stop = False
    for i, k in enumerate(order):                                   # Holm step-down at 0.05
        thr = 0.05 / (len(order) - i)
        sec[k]["holm_threshold"] = thr
        sec[k]["holm_reject"] = (not stop) and sec[k]["p_exact"] <= thr
        stop = stop or not sec[k]["holm_reject"]
    out["secondary"] = sec
    out["per_task"] = {t: contrast([r for r in rows if r["task"] == t], arm("cobs+g2f"), arm("cobs"))
                       for t in ("multikey", "single_essay")}
    for g in ("straddle_g0", "straddle_g1"):
        for v in (True, False):
            sub = [r for r in rows if r["spans"] and r["spans"][0].get(g) is v]
            if sub:
                out.setdefault("by_straddle", {})[f"{g}={v}"] = dict(n=len(sub), **contrast(sub, arm("cobs+g2f"), arm("cobs")))
    out["dense_accuracy"] = sum(int(r["res"][arm("dense")]["strict"]) for r in rows) / len(rows)
    cont = {m: [int(score(r["task"], r["res"][arm(m)]["out"], r["answers"]) >= 1) for r in rows] for m in spec["methods"]}
    out["containment_descriptive"] = dict(
        per_arm={m: dict(count=sum(v), n=len(v), rate=sum(v) / len(v)) for m, v in cont.items()},
        pairs={k: dict(improve=sum(x and not y for x, y in zip(cont[f"{k}+g2f"], cont[k])),
                       worsen=sum(y and not x for x, y in zip(cont[f"{k}+g2f"], cont[k])))
               for k in ("cobs", "nf_pt1", "quest")},
        note="descriptive secondary endpoint; no test, not part of any multiplicity family")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
