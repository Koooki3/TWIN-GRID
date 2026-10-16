"""Box-summary P-grid (exploratory, descriptive; reviewer batch 10, E4). Written and committed BEFORE the run starts.
Same 7 unit tasks x 20 frozen lines, Qwen2.5-1.5B, 16k, budget 512, U = 32 as P-grid. Arms: quest16 (B16 single
grid, G3-matched) and quest32+g2f (B32 TWIN-GRID) from runs/ruler/quest_pgrid, quest32 (B32 single) from the variants
run (runs/ruler/variants_pred/sum). Summary memory: quest16 = quest32+g2f = 16 floats/token (equal); quest32 = 8.
Purpose: separate the two side benefits of a finer grid (reviewer E2). A box summary is insensitive to the unit
fraction f, so
  - if quest16 still clearly beats quest32+g2f on the 4-target tasks (niah_multivalue, niah_multiquery), the
    selectable-position effect is the main driver;
  - if they are close there, the f effect (which a box summary does not have) explains the moment-summary P-grid gap.
No decision rule. A sample enters only if all three arms have it.
Reports: unit accuracy per arm overall / per task / single-target (niah_single_1/2, niah_multikey_1/2) vs 4-target
(niah_multivalue, niah_multiquery) vs vt; split by the 16-grid (X0@16) and by the 32-grid class; differences
quest32+g2f - quest16 with a sample-level bootstrap within task (2000).
Usage: quest_pgrid_analysis.py DATA_DIR QUEST_PGRID_DIR VARIANTS_SUM_DIR TOKENIZER [N]"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cluster"))
from aggregate import postprocess_pred  # noqa: E402
from unit_lengths import measure  # noqa: E402

TASKS = ("niah_single_1", "niah_single_2", "niah_multikey_1", "niah_multikey_2", "niah_multivalue", "niah_multiquery",
         "vt")
GROUPS = {"single-target": ("niah_single_1", "niah_single_2", "niah_multikey_1", "niah_multikey_2"),
          "4-target": ("niah_multivalue", "niah_multiquery"), "vt": ("vt",)}
ARMS = ("quest16", "quest32", "quest32+g2f")


def load(path):
    return {json.loads(l)["sample_uid"]: json.loads(l) for l in open(path)} if os.path.exists(path) else {}


def main(data, qp, vs, tokdir, n):
    u16 = measure(data, tokdir, n, 16, TASKS, 64, 32)
    u32 = measure(data, tokdir, n, 32, TASKS, 64, 32)
    recs = []
    for t in TASKS:
        lines = [json.loads(l) for l in open(os.path.join(data, t, "validation.jsonl"))][:n]
        rows = {"quest16": load(f"{qp}/b16/quest@512/{t}.jsonl"), "quest32+g2f": load(f"{qp}/b32/quest+g2f@512/{t}.jsonl"),
                "quest32": load(f"{vs}/quest@512/{t}.jsonl")}
        for i, ln in enumerate(lines):
            uid = next((u for u in rows["quest16"] if u.endswith(f":{t}:{i}")), None)
            if uid is None or any(uid not in rows[a] for a in ARMS):
                continue
            for k, ref in enumerate(ln["outputs"]):
                recs.append(dict(task=t, i=i, split16=bool(u16["tasks"][t]["samples"][i]["x0"][k]),
                                 cls=u32["tasks"][t]["samples"][i]["cls"][k],
                                 ok={a: int(ref.lower() in postprocess_pred(rows[a][uid]["pred"]).lower()) for a in ARMS}))
    samples = sorted({(r["task"], r["i"]) for r in recs})
    print(f"box-summary P-grid (exploratory): {len(recs)} units, {len(samples)} samples (of {len(TASKS) * n})")
    by = {s: [r for r in recs if (r["task"], r["i"]) == s] for s in samples}
    rng = np.random.default_rng(0)
    picks = []
    for _ in range(2000):
        p = []
        for t in TASKS:
            ss = [s for s in samples if s[0] == t]
            p += [ss[j] for j in rng.integers(0, len(ss), len(ss))] if ss else []
        picks.append(p)

    def acc(u, a):
        return np.mean([r["ok"][a] for r in u]) if u else np.nan

    def line(name, filt):
        u = [r for r in recs if filt(r)]
        if not u:
            return
        d = [acc([r for s in p for r in by[s] if filt(r)], "quest32+g2f") - acc([r for s in p for r in by[s] if filt(r)],
                                                                                   "quest16") for p in picks]
        d = [x for x in d if np.isfinite(x)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {name:30s} units {len(u):4d}: " + "  ".join(f"{a} {acc(u, a):.3f}" for a in ARMS)
              + f"  | g2f@32 - @16 {acc(u, 'quest32+g2f') - acc(u, 'quest16'):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    line("all", lambda r: True)
    for g, ts in GROUPS.items():
        line(g, lambda r, ts=ts: r["task"] in ts)
    for t in TASKS:
        line(t, lambda r, t=t: r["task"] == t)
    line("split by 16-grid", lambda r: r["split16"])
    line("not split by 16-grid", lambda r: not r["split16"])
    for c in ("in0", "repair"):
        line(f"32-grid class {c}", lambda r, c=c: r["cls"] == c)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 20)
