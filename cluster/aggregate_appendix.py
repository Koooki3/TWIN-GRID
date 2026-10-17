"""v1.4 APPENDIX aggregation (descriptive only; docs/PLAN_V14_APPENDIX_ZH.md; reviewer batch 11, F2).
Runs the frozen cluster/aggregate.py unchanged — same gate (RULER sources, frozen data and uids, manifests and protocol,
per-row hashes, budget counters, token_sha agreement) and the same official scoring — after adding, IN MEMORY ONLY,
the appendix arms (mean, mean+g2f, delim_cobs) to its arm table and emptying its primary pair and secondary family, so
every appendix comparison is labelled DESCRIPTIVE and carries no test. cluster/plan_v1.json on disk is not modified
(the JSON output still records its sha256). Use the appendix run tag written by cluster/make_appendix_jobs.py.
Usage: aggregate_appendix.py <cluster/aggregate.py arguments>   e.g. --arm-a mean+g2f --arm-b quest32 --run-tag t3_n100_appendix-sum"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import aggregate  # noqa: E402

APPENDIX_ARMS = {"mean": dict(method="mean", block=32, status="ready (appendix)"),
                 "mean+g2f": dict(method="mean+g2f", block=32, status="ready (appendix)"),
                 "delim_cobs": dict(method="delim_cobs", block=32, status="ready (appendix; src/delim_dev.py runner)")}
aggregate.PLAN["arms"].update(APPENDIX_ARMS)
aggregate.PLAN["primary_pair"] = []           # appendix: nothing is PRIMARY
aggregate.PLAN["secondary_family"] = []       # appendix: nothing is SECONDARY

if __name__ == "__main__":
    tag = sys.argv[sys.argv.index("--run-tag") + 1] if "--run-tag" in sys.argv else ""
    if "_appendix-" not in tag:
        raise SystemExit("refusing: appendix aggregation needs an appendix run tag (…_appendix-sum / …_appendix-delim)")
    aggregate.main()
