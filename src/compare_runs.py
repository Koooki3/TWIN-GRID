"""Paired comparison across runs that share seeds: python compare_runs.py REF=file:method@bud A=file:method@bud ..."""
import json
import sys

import numpy as np

from analyze import boot


def load(spec):
    path, key = spec.rsplit(":", 1)
    return {(r["task"], r["seed"]): r["res"][key]["score"] for r in map(json.loads, open(path)) if key in r["res"]}


def main(args):
    name0, spec0 = args[0].split("=", 1)
    ref = load(spec0)
    print(f"reference {name0}: {np.mean(list(ref.values())):.3f} (n={len(ref)})")
    for a in args[1:]:
        name, spec = a.split("=", 1)
        x = load(spec)
        keys = sorted(set(ref) & set(x))
        d = np.array([x[k] - ref[k] for k in keys])
        mu, lo, hi = boot(d)
        print(f"{name:14s} {np.mean([x[k] for k in keys]):.3f}  − {name0}: {mu:+.3f} [{lo:+.3f},{hi:+.3f}] (n={len(keys)})")


if __name__ == "__main__":
    main(sys.argv[1:])
