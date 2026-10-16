"""Independent recomputation of the headline development numbers (variants and box-summary P-grid), standard library
only, without project code (no aggregate.py / unit_lengths.py): reads the frozen data's answer references and each arm's
prediction file, applies a post-processing rule inline (strip; control characters -> newline; newlines -> spaces; a
substring match is unaffected by the last step) and a case-insensitive substring match per reference, and prints the
overall unit accuracy per arm on samples that every arm has. Used as a self-check while the third-party reviewer
session was offline (2026-10-17); results/recompute_dev_independent.log."""
import json
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
DATA = os.path.join(ROOT, "runs/ruler/pgrid_qwen1.5b_16k")
TASKS = ("niah_single_1", "niah_single_2", "niah_multikey_1", "niah_multikey_2", "niah_multivalue", "niah_multiquery",
         "vt")
ARMS = {"cobs": "runs/ruler/pgrid_pred/b32/cobs@512", "cobs+g2f": "runs/ruler/pgrid_pred/b32/cobs+g2f@512",
        "cobs@16": "runs/ruler/pgrid_pred/b16/cobs@512", "cobs+nbh": "runs/ruler/variants_pred/nb/cobs+nbh@512",
        "mean+g2f": "runs/ruler/variants_pred/sum/mean+g2f@512", "quest32": "runs/ruler/variants_pred/sum/quest@512",
        "quest16": "runs/ruler/quest_pgrid/b16/quest@512", "quest32+g2f": "runs/ruler/quest_pgrid/b32/quest+g2f@512"}


def post(s):
    s = s.strip()
    s = re.sub(r"[\x00-\x1f]", "\n", s)
    return s.replace("\n", " ").strip()


tot = {a: [0, 0] for a in ARMS}
for t in TASKS:
    refs = [json.loads(l)["outputs"] for l in open(os.path.join(DATA, t, "validation.jsonl"))][:20]
    preds = {}
    for a, d in ARMS.items():
        preds[a] = {int(json.loads(l)["sample_uid"].rsplit(":", 1)[1]): post(json.loads(l)["pred"]).lower()
                    for l in open(os.path.join(ROOT, d, f"{t}.jsonl"))}
    common = set.intersection(*[set(p) for p in preds.values()])
    for i in sorted(common):
        for a in ARMS:
            for r in refs[i]:
                tot[a][0] += r.lower() in preds[a][i]
                tot[a][1] += 1
for a, (k, n) in tot.items():
    print(f"{a:12s} {k:4d}/{n} = {k / n:.3f}")
