"""P-grid development analysis (exploratory; plan v1.4 P-grid, theory L4). Local Qwen2.5-1.5B, official RULER data,
predictions from scripts/queue_pgrid.sh. Per unit (cluster/unit_lengths.py v3, one unit per answer reference) the
16-grid straddle X0@16 (block 16, layout unit 32) and the 32-grid class (block 32) are measured on the same prompt.
A unit is answered iff its reference occurs in the post-processed prediction (RULER string_match_all credit).
Table per scorer s in {cobs, nf_pt1}: unit accuracy of s@16 (single fine grid), s@32 (single coarse grid) and s+g2f@32
(TWIN-GRID), split by "split by the 16-grid" (X0@16) and "not split"; paired counts s+g2f vs s@16 on split units
(L4 predicts twin >= fine there). Intervals: sample-level bootstrap within task (2000). Development data, 20 samples per
task — exploratory only; no decision rule.
Usage: pgrid_dev.py DATA_DIR PRED_ROOT TOKENIZER [N]"""
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


def main(data, pred_root, tokdir, n):
    u16 = measure(data, tokdir, n, 16, TASKS, 64, 32)
    u32 = measure(data, tokdir, n, 32, TASKS, 64, 32)
    recs = []                                     # (task, sample, split16, cls32, {arm: ok})
    for t in TASKS:
        lines = [json.loads(l) for l in open(os.path.join(data, t, "validation.jsonl"))][:n]
        rows = {}
        for arm, sub in (("cobs@16", "b16/cobs@512"), ("nf_pt1@16", "b16/nf_pt1@512"), ("cobs", "b32/cobs@512"),
                         ("cobs+g2f", "b32/cobs+g2f@512"), ("nf_pt1", "b32/nf_pt1@512"),
                         ("nf_pt1+g2f", "b32/nf_pt1+g2f@512")):
            p = os.path.join(pred_root, sub, f"{t}.jsonl")
            rows[arm] = [json.loads(l) for l in open(p)] if os.path.exists(p) else []
        for i, ln in enumerate(lines):
            preds = {}
            for arm, rr in rows.items():
                hit = [r for r in rr if r["sample_uid"].endswith(f":{t}:{i}")]
                if hit:
                    preds[arm] = postprocess_pred(hit[0]["pred"]).lower()
            if len(preds) < 6:
                continue
            s16, s32 = u16["tasks"][t]["samples"][i], u32["tasks"][t]["samples"][i]
            for k, ref in enumerate(ln["outputs"]):
                recs.append((t, i, bool(s16["x0"][k]), s32["cls"][k],
                             {arm: int(ref.lower() in p) for arm, p in preds.items()}))
    print(f"P-grid development (exploratory): {len(recs)} units, {len({(r[0], r[1]) for r in recs})} samples, "
          f"tasks {sorted({r[0] for r in recs})}")
    rng = np.random.default_rng(0)
    samples = sorted({(r[0], r[1]) for r in recs})
    by_sample = {s: [r for r in recs if (r[0], r[1]) == s] for s in samples}
    for sc in ("cobs", "nf_pt1"):
        for split in (True, False):
            sel = [r for r in recs if r[2] == split]
            if not sel:
                continue
            acc = {a: np.mean([r[4][a] for r in sel]) for a in (f"{sc}@16", sc, f"{sc}+g2f")}
            win = sum(1 for r in sel if r[4][f"{sc}+g2f"] and not r[4][f"{sc}@16"])
            loss = sum(1 for r in sel if r[4][f"{sc}@16"] and not r[4][f"{sc}+g2f"])
            diffs = []
            for _ in range(2000):
                pick = []
                for t in sorted({s[0] for s in samples}):
                    ss = [s for s in samples if s[0] == t]
                    pick += [ss[j] for j in rng.integers(0, len(ss), len(ss))]
                u = [r for s in pick for r in by_sample[s] if r[2] == split]
                if u:
                    diffs.append(np.mean([r[4][f"{sc}+g2f"] - r[4][f"{sc}@16"] for r in u]))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            print(f"  {sc:6s} {'split by 16-grid' if split else 'not split      '} units {len(sel):4d}: "
                  f"{sc}@16 {acc[f'{sc}@16']:.3f}  {sc}@32 {acc[sc]:.3f}  {sc}+g2f@32 {acc[f'{sc}+g2f']:.3f}; "
                  f"g2f vs @16 +{win}/-{loss}, diff {acc[f'{sc}+g2f'] - acc[f'{sc}@16']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for sc in ("cobs", "nf_pt1"):
        for c in ("in0", "repair", "out", "fixed"):
            sel = [r for r in recs if r[3] == c]
            if sel:
                print(f"  {sc:6s} 32-grid class {c:6s} units {len(sel):4d}: {sc}@32 {np.mean([r[4][sc] for r in sel]):.3f}"
                      f"  {sc}+g2f {np.mean([r[4][sc + '+g2f'] for r in sel]):.3f}  {sc}@16 {np.mean([r[4][sc + '@16'] for r in sel]):.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 20)
