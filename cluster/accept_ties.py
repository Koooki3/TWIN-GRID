"""Final adapter-acceptance verdict under the bf16 tie rule (plan v1.4 §3.2; reviewer batch 3 conditions a-d).
Input: the strict report of cluster/accept_adapter.py and one cluster/accept_tie_diag.py JSON per same-input mismatch.
ACCEPTED_WITH_TIES only if: every order / cache-isolation step passed; every same-input mismatch has a diagnostic with
tie = true (input tokens equal, top-2 swap at the first divergence, margin <= 2 bf16 ulps) and equal official scores;
at most 1 tie per task; ties <= floor(0.02 * same-input rows). Otherwise NOT ACCEPTED. A strict pass stays ACCEPTED.
Usage: accept_ties.py REPORT.json [DIAG.json ...]"""
import json
import math
import sys


def main(report, diags):
    r = json.load(open(report))
    if r.get("input_problems"):
        print("NOT ACCEPTED: input validation failed")
        return 1
    steps = r["steps"]
    order_ok = all(v["status"] == "identical" and min(v["covered_rows"].values()) > 0
                   for k, v in steps.items() if k.startswith("order:"))
    rows = [(k.split(":", 1)[1], row) for k, v in steps.items() if k.startswith("same_input:") for row in v["rows"]]
    mism = [(t, row) for t, row in rows if not row["same_text"]]
    d = {x["uid"]: x for x in (json.load(open(p)) for p in diags)}
    problems, per_task = [], {}
    for t, row in mism:
        x = d.get(row["uid"])
        if x is None or not x.get("tie"):
            problems.append(f"{row['uid']}: no tie diagnostic or not a tie")
        if row["score_ref"] != row["score_ours"]:
            problems.append(f"{row['uid']}: official scores differ")
        per_task[t] = per_task.get(t, 0) + 1
    problems += [f"{t}: {c} ties (> 1 per task)" for t, c in per_task.items() if c > 1]
    cap = math.floor(0.02 * len(rows))
    if len(mism) > cap:
        problems.append(f"{len(mism)} ties > cap floor(0.02 * {len(rows)}) = {cap}")
    if not order_ok:
        problems.append("order / cache-isolation step failed")
    if not mism and order_ok:
        print("ACCEPTED (strict)")
        return 0
    if problems:
        print("NOT ACCEPTED:", *problems, sep="\n  ")
        return 1
    print(f"ACCEPTED_WITH_TIES: {len(mism)} bf16 tie(s) of {len(rows)} same-input rows: "
          + ", ".join(u["uid"] for _, u in mism))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
