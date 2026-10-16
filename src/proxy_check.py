"""Does the cheap offline statistic (target-needle mass retained at the first decode step)
predict end-to-end accuracy? If yes, longer contexts can be screened with one decode step."""
import json
import sys

import numpy as np

METHODS = ["o_mass", "quest", "mean", "cobs", "sum_dual", "sk_mass", "sk_peak", "sk_dual"]


def main(path):
    recs = [json.loads(l) for l in open(path)]
    recs = [r for r in recs if r["task"] != "single_noise"]          # saturated task
    for b in ("256", "512"):
        rows, xs, ys, zs = [], [], [], []
        for m in METHODS:
            acc, ret, low = [], [], []
            for r in recs:
                d = r["stats"][b][m]
                nk = sum(v[2] for v in d.values())
                nt = sum(v[3] for v in d.values())
                per_layer = [v[2] / v[3] for v in d.values() if v[3] > 1e-3]
                acc.append(r["res"][f"{m}@{b}"]["score"])
                ret.append(nk / nt)
                low.append(min(per_layer))
            xs += ret; ys += acc; zs += low
            rows.append((m, np.mean(acc), np.mean(ret), np.mean(low)))
        A = np.array([r[1:] for r in rows])
        print(f"budget {b}: per-sample corr(acc, retention) {np.corrcoef(xs, ys)[0, 1]:.2f}, "
              f"corr(acc, min-layer retention) {np.corrcoef(zs, ys)[0, 1]:.2f}; "
              f"method-level {np.corrcoef(A[:, 0], A[:, 1])[0, 1]:.2f} / {np.corrcoef(A[:, 0], A[:, 2])[0, 1]:.2f}")
        for m, a, r, l in rows:
            print(f"   {m:9s} acc {a:.2f}  retention {r:.3f}  min-layer {l:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
