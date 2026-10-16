"""Paired comparison of variants against a reference method, plus a reproducibility check
against an earlier run with the same seeds."""
import argparse
import json

import numpy as np

from analyze import boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref", default="sum_dual")
    ap.add_argument("--prev", default=None, help="earlier JSONL to check identical scores")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.run)]
    keys = sorted({k for r in recs for k in r["res"]})
    budgets = sorted({int(k.split("@")[1]) for k in keys})
    methods = list(dict.fromkeys(k.split("@")[0] for k in recs[0]["res"]))
    tasks = sorted({r["task"] for r in recs})
    for bud in budgets:
        print(f"### budget={bud} (n={len(recs)}: " + ", ".join(
            f"{t}={sum(r['task'] == t for r in recs)}" for t in tasks) + ")\n")
        print("| method | " + " | ".join(tasks) + f" | all | − {a.ref} [95% CI] |")
        print("|---" * (len(tasks) + 3) + "|")
        ref = np.array([r["res"][f"{a.ref}@{bud}"]["score"] for r in recs])
        for m in methods:
            if m == "dense":
                continue
            sc = np.array([r["res"][f"{m}@{bud}"]["score"] for r in recs])
            cells = [f"{np.mean([s for s, r in zip(sc, recs) if r['task'] == t]):.2f}" for t in tasks]
            mu, lo, hi = boot(sc - ref)
            print(f"| {m} | " + " | ".join(cells) + f" | {sc.mean():.3f} | {mu:+.3f} [{lo:+.3f},{hi:+.3f}] |")
        print()
    if a.prev:
        prev = {(r["task"], r["ctx"], r["seed"]): r for r in map(json.loads, open(a.prev))}
        same = diff = 0
        for r in recs:
            p = prev.get((r["task"], r["ctx"], r["seed"]))
            for k, v in r["res"].items():
                if p and k in p["res"]:
                    if p["res"][k]["out"] == v["out"]:
                        same += 1
                    else:
                        diff += 1
        print(f"Reproducibility vs {a.prev}: identical outputs {same}, different {diff}")


if __name__ == "__main__":
    main()
