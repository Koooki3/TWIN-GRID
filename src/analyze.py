"""Summarize stage-1 JSONL: accuracy (bootstrap 95% CI), paired gate tests, offline stats."""
import json
import sys
from collections import defaultdict

import numpy as np

FAMILIES = {"oracle": ("o_mass", "o_peak", "o_dual"),
            "sketch": ("sk_mass", "sk_peak", "sk_dual"),
            "summary": ("cobs", "quest", "sum_dual")}
ORDER = ["dense", "o_mass", "o_peak", "o_beta2", "o_dual", "quest", "mean", "cobs", "sum_dual",
         "sk_mass", "sk_peak", "sk_beta2", "sk_dual"]


def boot(x, B=4000, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main(paths):
    recs = [json.loads(l) for p in paths for l in open(p)]
    by = defaultdict(list)
    for r in recs:
        by[(r["ctx"], r["task"])].append(r)
    keys = sorted({k for r in recs for k in r["res"]})
    budgets = sorted({int(k.split("@")[1]) for k in keys})
    found = {k.split("@")[0] for k in keys}
    methods = [m for m in ORDER if m in found] + sorted(found - set(ORDER))

    print("## Accuracy (mean, 95% bootstrap CI over samples)\n")
    for ctx in sorted({c for c, _ in by}):
        allr = [r for (c, _), v in by.items() if c == ctx for r in v]
        tasks = sorted(t for c, t in by if c == ctx)
        for bud in budgets:
            print(f"### ctx={ctx} budget={bud}  (n: " +
                  ", ".join(f"{t}={len(by[(ctx, t)])}" for t in tasks) + ")\n")
            print("| method | " + " | ".join(tasks) + " | all |")
            print("|---" * (len(tasks) + 2) + "|")
            for m in methods:
                k = f"dense@{budgets[0]}" if m == "dense" else f"{m}@{bud}"
                cells = []
                for t in tasks + ["all"]:
                    rs = allr if t == "all" else by[(ctx, t)]
                    mu, lo, hi = boot([r["res"][k]["score"] for r in rs if k in r["res"]])
                    cells.append(f"{mu:.2f} [{lo:.2f},{hi:.2f}]")
                print(f"| {m} | " + " | ".join(cells) + " |")
            print(f"\nGate (paired over all samples, ctx={ctx}, budget={bud}): dual − best single channel")
            for fam, (ma, pk, du) in FAMILIES.items():
                if not all(f"{x}@{bud}" in allr[0]["res"] for x in (ma, pk, du)):
                    continue
                sc = {m: np.array([r["res"][f"{m}@{bud}"]["score"] for r in allr]) for m in (ma, pk, du)}
                best = ma if sc[ma].mean() >= sc[pk].mean() else pk
                mu, lo, hi = boot(sc[du] - sc[best])
                print(f"- {fam}: dual={sc[du].mean():.3f} mass={sc[ma].mean():.3f} peak={sc[pk].mean():.3f}; "
                      f"dual−{best}={mu:+.3f} [{lo:+.3f},{hi:+.3f}]")
            print()

    print("## Offline stats at the first decode step (layers >= 2)\n")
    for ctx in sorted({c for c, _ in by}):
        allr = [r for (c, _), v in by.items() if c == ctx for r in v]
        for bud in budgets:
            print(f"### ctx={ctx} budget={bud}\n")
            print("| method | needle-mass retained | mass captured | rel. output error |")
            print("|---|---|---|---|")
            for m in methods:
                if m == "dense":
                    continue
                nk = nt = 0.0
                ms, es = [], []
                for r in allr:
                    d = r["stats"][str(bud)].get(m)
                    for v in (d or {}).values():
                        ms.append(v[0]); es.append(v[1]); nk += v[2]; nt += v[3]
                if ms:
                    print(f"| {m} | {nk / max(nt, 1e-9):.3f} | {np.mean(ms):.4f} | {np.mean(es):.4f} |")
            print()


if __name__ == "__main__":
    main(sys.argv[1:])
