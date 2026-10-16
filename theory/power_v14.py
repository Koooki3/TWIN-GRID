"""Pre-freeze power for plan v1.4 (reviewer batch-1 correction 2): primary = cobs+g2f vs cobs, 13-task macro average,
n = 100 per task; Holm family (2 members, batch-2 consensus) includes the same pair on the coverable tasks. Scenario "gain only on coverable tasks", with
the crossing indicators of each unit MEASURED by cluster/unit_lengths.py (JSON) on dev data prepared with the primary
model's tokenizer.

Outcome model per unit by its unit_lengths v2 class (local C3, cobs, 300 prompts; results/phase2_kv_final.log K3):
"repair" (straddles grid 0, inside a grid-1 block) -> +1 with prob g, -1 with prob 0.01; "in0" -> +1 0.015, -1 0.025
(local 3/201 and 5/201); "out" (straddles, not recoverable: R = 0) -> +1/-1 with prob 0.01 each; "fixed" -> 0. g = 0.40 locally (40/99);
scenarios also g = 0.20 and 0.10 (an 8B model may fail less often on cut units). Multi-unit tasks (string_match_all,
per-unit credit): sample diff = mean of its unit diffs (independent units: optimistic for vt chains). The same class
model applies to every task with units (not-coverable UUID tasks are mostly "out"); no-unit tasks: symmetric discordance (+1/-1 with prob 0.03 each, 0.05 for qa_*, cwe/fwe continuous N(0, 0.1^2)) —
no effect, only variance. Test: stratified sign-flip permutation, normal approximation of its null (var = sum d^2 /
n^2 / T^2), two-sided; power at alpha 0.05 (primary) and at 0.025 (worst step of the two-member Holm family).
Usage: power_v14.py UNIT_LENGTHS.json [R] [g1,g2,...]   (the last form scans for the minimum detectable g)"""
import json
import sys

import numpy as np

res = json.load(open(sys.argv[1]))
R = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
N = 100
rng = np.random.default_rng(0)
Z = {0.05: 1.959964, 0.025: 2.241403}
tasks = res["tasks"]
cov = [t for t, r in tasks.items() if r["group"] == "coverable"]
print(f"tokenizer {res['tokenizer_json_sha256'][:16]}; dev samples/task {len(next(iter(tasks.values()))['samples'])}; "
      f"coverable ({len(cov)}): {', '.join(cov)}")
for t, r in tasks.items():
    if r["group"] != "no_unit":
        u = [c for s in r["samples"] for c in s["cls"]]
        ls = [x for s in r["samples"] for x in s["lens"]]
        print(f"  {t:16s} {r['group']:13s} unit length {min(ls)}-{max(ls)} mean {np.mean(ls):.1f}; pi(repair) {u.count('repair')/len(u):.3f}"
              f"; out {u.count('out')/len(u):.3f}")


def sim_task(r, g, size):
    """size x N matrix of per-sample paired differences for one task."""
    if r["group"] != "no_unit":
        smp = r["samples"]
        out = np.zeros((size, N))
        for j in range(N):
            pick = rng.integers(0, len(smp), size=size)          # resample measured samples (crossing patterns)
            for k, i in enumerate(pick):
                cl = smp[i]["cls"]
                u = rng.random(len(cl))
                pg = np.array([{"repair": g, "in0": 0.015, "out": 0.01, "fixed": 0.0}[c] for c in cl])
                pl = np.array([{"repair": 0.01, "in0": 0.025, "out": 0.01, "fixed": 0.0}[c] for c in cl])
                out[k, j] = np.mean((u < pg) * 1.0 - ((u >= pg) & (u < pg + pl)) * 1.0)
        return out
    t = r["_name"]
    if t in ("cwe", "fwe"):
        return rng.normal(0, 0.1, size=(size, N))
    q = 0.05 if t.startswith("qa") else 0.03
    u = rng.random((size, N))
    return (u < q) * 1.0 - ((u >= q) & (u < 2 * q)) * 1.0


for t in tasks:
    tasks[t]["_name"] = t
GS = [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else (0.40, 0.20, 0.10)
for g in GS:
    sims = {t: sim_task(r, g, R) for t, r in tasks.items()}
    for label, ts in (("macro 13 tasks (primary)", list(tasks)), (f"coverable {len(cov)} tasks (Holm member)", cov)):
        d = np.stack([sims[t] for t in ts])                       # T x R x N
        D = d.mean(2).mean(0)
        se = np.sqrt((d ** 2).sum(2).sum(0)) / (N * len(ts))
        z = np.abs(D) / np.maximum(se, 1e-12)
        pw = {a: float(np.mean(z > c)) for a, c in Z.items()}
        print(f"g={g:.2f}  {label:36s} E[D] {100 * D.mean():+5.2f} pts (sd {100 * D.std():.2f}); "
              f"power a=0.05 {pw[0.05]:.3f}, a=0.025 {pw[0.025]:.3f}")
