"""Phase-1 (development) analysis for prereg draft v2 — descriptive, no significance claims.
Hard integrity gate first (exit 2 on any failure); with --partial only integrity/progress is reported and NO
planning is done.
1. Integrity: exactly the pre-registered keys (seeds 110000-110039 multikey, 120000-120039 single_essay), exactly
   the seven arms, one code identity, identical argv, manifest without legacy rows / code changes, strict in {0,1},
   calls > 0 and budget_checked_calls == calls, all violation counters and max error zero.
2. Strict accuracy per arm and task; paired 2x2 tables for X+g2f vs X.
3. Natural crossing rate (value straddles grid 0).
4. Planning for the primary contrast (cobs+g2f vs cobs), reviewer's formula:
   M ~ Bin(N, d), G | M = m ~ Bin(m, pi), pi = (d + Delta) / (2 d); power(N) = sum_m P(M = m) P(reject | m, pi),
   exact two-sided McNemar at alpha 0.05. N = TOTAL prompts (two tasks, equal numbers; pooled homogeneous planning
   model). The reported N is the exact minimum over EVEN N in [20, 600]. Scenarios with d < |Delta| are impossible.
Usage: phase1_analysis.py PHASE1.jsonl [--partial]"""
import json
import sys
from collections import Counter

from scipy.stats import binom

ARMS = ["cobs", "cobs+g2f", "nf_pt1", "nf_pt1+g2f", "quest", "quest+g2f"]
TASKS = {"multikey": 110000, "single_essay": 120000}
N_PER_TASK = 40


def gate(path, rows, partial):
    problems = []
    keys = Counter((r["task"], r["ctx"], r["seed"]) for r in rows)
    if any(v > 1 for v in keys.values()):
        problems.append("duplicate keys")
    want = {(t, 16384, base + i) for t, base in TASKS.items() for i in range(N_PER_TASK)}
    extra = set(keys) - want
    if extra:
        problems.append(f"{len(extra)} keys outside the pre-registered set")
    missing = want - set(keys)
    if missing and not partial:
        problems.append(f"{len(missing)} pre-registered keys missing")
    if len({json.dumps(r["run_id"]["code"], sort_keys=True) for r in rows}) != 1:
        problems.append("more than one code identity")
    if len({json.dumps([x for x in r["run_id"]["argv"] if x not in ("--allow-code-change", "--adopt-legacy")])
            for r in rows}) != 1:
        problems.append("more than one argv")
    man = json.load(open(path + ".manifest.json"))
    if man.get("legacy_rows_before_manifest") or man.get("code_changes") or man.get("dev_flags"):
        problems.append("manifest has legacy rows, code changes or dev flags")
    for r in rows:
        if set(k.split("@")[0] for k in r["res"]) != set(ARMS) | {"dense"}:
            problems.append(f"arm set differs at {r['task']} {r['seed']}")
            break
        for k, v in r["res"].items():
            if v.get("strict") not in (0, 0.0, 1, 1.0):
                problems.append(f"strict not binary at {r['seed']} {k}")
            if k.startswith("dense"):
                continue
            t = v.get("tokens") or {}
            if not t.get("calls") or t.get("budget_checked_calls") != t.get("calls") \
                    or t.get("budget_violations") or t.get("max_abs_budget_error") or t.get("fixed_region_violations"):
                problems.append(f"budget counters at {r['seed']} {k}: {t}")
    return problems, missing


_REJ = {}


def rejection_prob(m, pi, alpha=0.05):
    """P(exact two-sided McNemar rejects | m discordant pairs, improvement probability pi)."""
    if (m, alpha) not in _REJ:                     # cache keyed by (m, alpha) (Codex B1)
        rej = [g for g in range(m + 1) if min(1.0, 2 * binom.cdf(min(g, m - g), m, 0.5)) <= alpha]
        _REJ[(m, alpha)] = rej
    return sum(binom.pmf(g, m, pi) for g in _REJ[(m, alpha)])


def power(N, d, delta, alpha=0.05):
    pi = (d + delta) / (2 * d)
    return sum(binom.pmf(m, N, d) * rejection_prob(m, pi, alpha) for m in range(N + 1) if binom.pmf(m, N, d) > 1e-12)


def main(path, partial):
    rows = [json.loads(l) for l in open(path)]
    problems, missing = gate(path, rows, partial)
    print(f"rows {len(rows)}; per task {dict(Counter(r['task'] for r in rows))}; missing {len(missing)}")
    if problems:
        print("INTEGRITY GATE FAILED:")
        for p in problems[:20]:
            print("  -", p)
        sys.exit(2)
    print("integrity gate passed" + (" (partial file: progress only, no planning)" if partial else ""))
    if partial:
        return
    nat = sum(1 for r in rows if r["spans"] and r["spans"][0]["straddle_g0"])
    print(f"natural crossing (grid 0): {nat}/{len(rows)}")
    print("\n| arm | multikey | single_essay | all |  (strict)\n|---|---|---|---|")
    for a in ["dense"] + ARMS:
        k = f"{a}@512"
        by = {t: [r["res"][k]["strict"] for r in rows if r["task"] == t] for t in TASKS}
        allv = by["multikey"] + by["single_essay"]
        print(f"| {a} | {sum(by['multikey']):.0f}/{len(by['multikey'])} | {sum(by['single_essay']):.0f}/"
              f"{len(by['single_essay'])} | {sum(allv):.0f}/{len(allv)} |")
    print("\npaired (strict): X+g2f vs X — improve / worsen / both right / both wrong")
    disc = {}
    for base in ("cobs", "nf_pt1", "quest"):
        for t in ("multikey", "single_essay", "all"):
            c = Counter()
            for r in rows:
                if t != "all" and r["task"] != t:
                    continue
                x, y = r["res"][f"{base}@512"]["strict"], r["res"][f"{base}+g2f@512"]["strict"]
                c["improve" if y > x else "worsen" if y < x else ("both_right" if x else "both_wrong")] += 1
            print(f"  {base:7s} {t:12s} {dict(c)}")
            disc[(base, t)] = c
    c = disc[("cobs", "all")]
    n = sum(c.values())
    p10, p01 = c["improve"] / n, c["worsen"] / n
    d_obs = p10 + p01
    print(f"\nprimary pilot: p10 = {p10:.3f}, p01 = {p01:.3f}, discordance d = {d_obs:.3f} (n = {n}); per task: "
          + ", ".join(f"{t} d = {(disc[('cobs', t)]['improve'] + disc[('cobs', t)]['worsen']) / N_PER_TASK:.3f}"
                      for t in TASKS))
    print("\nplanning (exact minimum over even total N in [20, 600]; power of exact two-sided McNemar, alpha 0.05):")
    for delta in (0.08, 0.05):
        for scale in (1.0, 0.5, 1.5):
            d = d_obs * scale
            if d < delta or d > 1:
                print(f"  Delta {delta:.2f}, d {d:.3f} ({scale}x pilot): IMPOSSIBLE (needs |Delta| <= d <= 1)")
                continue
            need = next((N for N in range(20, 601, 2) if power(N, d, delta) >= 0.8), None)
            print(f"  Delta {delta:.2f}, d {d:.3f} ({scale}x pilot): minimum even N with power >= 0.8 = {need}; "
                  f"power at N = 600: {power(600, d, delta):.2f}")


if __name__ == "__main__":
    main(sys.argv[1], "--partial" in sys.argv[2:])
