"""Completeness gate for rep2 and its value appendix (reviewer, after the restart: rep2_analysis.units() silently skips
samples missing in any arm, so an incomplete arm would shrink the denominator instead of blocking the result).
src/rep2_analysis.py is registered and is NOT modified; this gate runs BEFORE it and its rc is quoted in the report.
Checks, per (arm, task, budget): b512 arms must contain exactly the uids of frozen lines 0..99, b1024 arms exactly
0..59, the value-appendix arms (em, quest16+g2f, quest8; b512) exactly 0..99 — no missing, no duplicate, no foreign uid
(uid = "<data sha16>:<task>:<line>" of runs/ruler/rep2_qwen1.5b_16k). Exit 0 if all complete; else list the gaps and
exit 2: every R/A item that uses an incomplete arm must then be reported "无法判定（不完整）", never computed on the
reduced set.
Also reports (reviewer point 3) the determinism split for the crashed-and-resumed cell mean+g2f / niah_multikey_2:
lines 0..13 (written before the crash, job 8de48c96) and 14..19 (written by the resume) are compared separately with
the round-1 mean+g2f predictions for the same samples.
Usage: rep2_gate.py [ROOT] [--value] [--b512-only]   (--value: also gate the three value-appendix arms;
       --b512-only: gate only the budget-512 rep2 arms, for an early R1–R6 verdict; R7 and A4/A5 then remain "not yet run")
Amendment 2026-10-18 (Codex T4, review of 70d0779): the success line used to say "b1024 lines 0..59" even under
--b512-only, where b1024 is not checked; it now names only the gated ranges. Check logic unchanged; the original
results/rep2_early.log of 70d0779 is kept as written."""
import hashlib
import json
import os
import sys

ROOT = os.path.abspath(next((x for x in sys.argv[1:] if not x.startswith("--")), "."))
DATA = os.path.join(ROOT, "runs/ruler/rep2_qwen1.5b_16k")
PRED = os.path.join(ROOT, "runs/ruler/rep2_pred")
TASKS = ("niah_single_1", "niah_single_2", "niah_multikey_1", "niah_multikey_2", "niah_multivalue", "niah_multiquery",
         "vt")
B512 = {"dense": "dense", "cobs": "cobs", "cobs+g2f": "cobs+g2f", "mean": "mean", "mean+g2f": "mean+g2f",
        "quest32": "quest", "quest32+g2f": "quest+g2f", "cobs16": "cobs", "quest16": "quest"}
B1024 = {"cobs": "cobs", "cobs+g2f": "cobs+g2f", "cobs16": "cobs", "quest32": "quest", "mean+g2f": "mean+g2f"}
VALUE = {"em": "em", "quest16+g2f": "quest+g2f", "quest8": "quest"}


def check(arms, budget, n, gaps):
    for arm, method in arms.items():
        for t in TASKS:
            sha = hashlib.sha256(open(os.path.join(DATA, t, "validation.jsonl"), "rb").read()).hexdigest()
            want = {f"{sha[:16]}:{t}:{i}" for i in range(n)}
            p = os.path.join(PRED, f"b{budget}", arm, f"{method}@{budget}", f"{t}.jsonl")
            uids = [json.loads(l)["sample_uid"] for l in open(p)] if os.path.exists(p) else []
            miss, extra, dup = want - set(uids), set(uids) - want, len(uids) - len(set(uids))
            if miss or extra or dup:
                gaps.append(f"b{budget} {arm:12s} {t:16s}: {len(uids)} rows, missing {len(miss)}, foreign {len(extra)}, "
                            f"duplicate {dup}")


def main():
    gaps = []
    check(B512, 512, 100, gaps)
    if "--b512-only" not in sys.argv:          # --b512-only: gate R1–R6 early; R7 and A4/A5 then stay "not yet run"
        check(B1024, 1024, 60, gaps)
    if "--value" in sys.argv:
        check(VALUE, 512, 100, gaps)
    # determinism split for the crashed cell (lines 0..13 before the crash, 14..19 after the resume)
    r1 = os.path.join(ROOT, "runs/ruler/variants_pred/sum/mean+g2f@512/niah_multikey_2.jsonl")
    r2 = os.path.join(PRED, "b512/mean+g2f/mean+g2f@512/niah_multikey_2.jsonl")
    if os.path.exists(r1) and os.path.exists(r2):
        a = {int(json.loads(l)["sample_uid"].rsplit(":", 1)[1]): json.loads(l)["pred"] for l in open(r1)}
        b = {int(json.loads(l)["sample_uid"].rsplit(":", 1)[1]): json.loads(l)["pred"] for l in open(r2)}
        for lo, hi, what in ((0, 14, "written before the crash"), (14, 20, "written by the resume")):
            ii = [i for i in range(lo, hi) if i in a and i in b]
            print(f"determinism mean+g2f/niah_multikey_2 lines {lo}..{hi - 1} ({what}): pred equal to round 1 "
                  f"{sum(a[i] == b[i] for i in ii)}/{len(ii)}")
    if gaps:
        print("GATE FAILED (incomplete arms; affected R/A items must be reported 无法判定（不完整）):", *gaps, sep="\n  ")
        sys.exit(2)
    # amendment 2026-10-18 (Codex T4): name only the ranges actually gated; the checks themselves are unchanged
    scope = ["b512 rep2 arms lines 0..99"]
    scope += ["b1024 lines NOT gated (R7 not judged)"] if "--b512-only" in sys.argv else ["b1024 lines 0..59"]
    scope += ["value arms lines 0..99"] if "--value" in sys.argv else ["value arms NOT gated (A4/A5 not judged)"]
    print("GATE PASSED: every gated arm complete (" + "; ".join(scope) + ")")


if __name__ == "__main__":
    main()
