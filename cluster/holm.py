"""Holm step-down (alpha 0.05) over the plan's secondary family in the primary cell (plan §3.4) — v2 after Codex Q3;
v1.4: the family has two members, one on the registered coverable task subset (pair ["cobs+g2f", "cobs", "coverable"]).
Inputs are cluster/aggregate.py --json outputs. Refused (exit 3) unless: exactly one result per family pair (duplicates
refused, extra pairs refused); every result has role SECONDARY, the current plan_sha256, the primary cell's
model/length/budget/n; all results share the same per-task data sha256, the same code/model/protocol identity; every
p is a finite number in [0, 1]. Results for the same frozen content may come from different run tags. Holm itself:
sort by p, threshold 0.05/(m-i), stop at the first non-rejection.
Usage: holm.py RESULT1.json RESULT2.json ..."""
import hashlib
import json
import math
import os
import sys

PLAN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan_v1.json")
PLAN = json.load(open(PLAN_PATH))


def refuse(msg):
    print(f"REFUSED: {msg}; Holm not computed; present members stay descriptive")
    sys.exit(3)


def main(paths):
    res = [json.load(open(p)) for p in paths]
    family = [tuple(x) for x in PLAN["secondary_family"]]
    pairs = [tuple(r.get("pair", [])) for r in res]
    if len(set(pairs)) != len(pairs):
        refuse("duplicate family members")
    if set(pairs) != set(family):
        refuse(f"members {sorted(pairs)} are not exactly the family {sorted(family)}")
    plan_sha = hashlib.sha256(open(PLAN_PATH, "rb").read()).hexdigest()
    pc = PLAN["primary_cell"]
    required = ("role", "plan_sha256", "model", "length", "budget", "n", "pair", "D", "ci95", "p", "data_sha256",
                "identity", "manifest_digest")
    for r in res:                                 # reviewer T6: identities must be present, not equal-because-missing
        missing = [k for k in required if r.get(k) in (None, "", {}, [])]
        if missing:
            refuse(f"{r.get('pair')}: missing {missing}")
        if set(r["data_sha256"]) != set(PLAN["tasks"]) or not (r["identity"].get("code") or {}):
            refuse(f"{r.get('pair')}: data fingerprints or code identity incomplete")
    for r in res:
        if r.get("role") != "SECONDARY" or r.get("plan_sha256") != plan_sha:
            refuse(f"{r.get('pair')}: not a SECONDARY result of the current plan")
        if (r.get("model"), r.get("length"), r.get("budget"), r.get("n")) != (pc["model"], pc["length"], pc["budget"],
                                                                               pc["samples"]):
            refuse(f"{r.get('pair')}: not the primary cell")
        p = r.get("p")
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not math.isfinite(p) or not 0 <= p <= 1:
            refuse(f"{r.get('pair')}: invalid p {p!r}")
    if len({json.dumps(r.get("data_sha256"), sort_keys=True) for r in res}) != 1:
        refuse("members were computed on different frozen data")
    if len({json.dumps(r.get("identity"), sort_keys=True) for r in res}) != 1:
        refuse("members have different code/model/protocol identities")
    got = {tuple(r["pair"]): r for r in res}
    order = sorted(family, key=lambda f: got[f]["p"])
    stop = False
    for i, f in enumerate(order):
        thr = 0.05 / (len(order) - i)
        rej = (not stop) and got[f]["p"] <= thr
        stop = stop or not rej
        name = f"{f[0]} vs {f[1]}" + (f" [{f[2]} tasks]" if len(f) > 2 else "")
        print(f"{name}: D {got[f]['D']:+.3f} [{got[f]['ci95'][0]:+.3f}, {got[f]['ci95'][1]:+.3f}] "
              f"p {got[f]['p']:.6f} threshold {thr:.4f} -> {'reject H0' if rej else 'do not reject'}")


if __name__ == "__main__":
    main(sys.argv[1:])
