"""Acceptance for the mechanism-diagnostic implementation check (mech design v2 + Codex M2). Exit 1 on failure.
Criteria (per prompt, per arm):
  - identity_checked and digit mapping 0..6 present;
  - same-arm TF vs FR agreement on every step before the first token divergence (and on the step predicting the
    first divergent token): |gold log-prob difference| <= TOL;
  - logging off/on gives identical tokens, reads and swap counts (logger_check);
  - every arm's per-call budget counters are zero and checked on every call;
  - forced arms: number of logged swaps equals the swap counter.
Completeness (Codex early feedback 1): exactly the pre-specified prompt keys (results/mech_prompts_check.json), no
duplicates, every row has exactly the four arms, non-empty TF/FR steps, non-empty token counters and a non-empty
logger_check; an empty or partial file is NOT ACCEPTED.
Usage: mech_check_accept.py MECH.jsonl [PROMPTS.json]"""
import json
import sys

TOL = 1e-3
ARMS = {"cobs", "nf_pt1", "quest", "cobs+ftail"}


def main(path, prompts):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    fails = []
    want = sorted(tuple(x) for x in json.load(open(prompts)))
    have = sorted((r["task"], r["seed"]) for r in rows)
    if have != want:
        fails.append(f"prompt keys {have} != pre-specified {want}")
    for r in rows:
        tag = f"{r['task']}:{r['seed']}"
        if set(r.get("arms", {})) != ARMS:
            fails.append(f"{tag}: arms {sorted(r.get('arms', {}))} != {sorted(ARMS)}")
        if not r.get("identity_checked"):
            fails.append(f"{tag}: identity not checked")
        if [d for d in r["digit_of"] if d is not None] != list(range(len(r["value"]))):
            fails.append(f"{tag}: digit mapping")
        for arm, v in r["arms"].items():
            tf, fr = v["tf"], v["fr"]
            if not tf["steps"] or not fr["steps"] or len(tf["steps"]) < len(fr["steps"]):
                fails.append(f"{tag} {arm}: empty or short TF/FR steps ({len(tf['steps'])}/{len(fr['steps'])})")
                continue
            fd = fr["first_divergence"]
            upto = len(fr["steps"]) if fd is None else min(fd + 1, len(fr["steps"]))
            for j in range(upto):
                a, b = tf["steps"][j], fr["steps"][j]
                if a["gold"] != b["gold"] or abs(a["gold_logprob"] - b["gold_logprob"]) > TOL:
                    fails.append(f"{tag} {arm}: TF/FR disagree at step {j} ({a['gold_logprob']:.4f} vs "
                                 f"{b['gold_logprob']:.4f})")
                    break
            lc = v.get("logger_check")
            if not lc or not all(lc.values()):
                fails.append(f"{tag} {arm}: logger check {lc}")
            for mode in ("tf", "fr"):
                if not v[mode]["tokens"]:
                    fails.append(f"{tag} {arm} {mode}: no budget counters")
                for m, t in v[mode]["tokens"].items():
                    # tokens = [min, max, sum, count, budget_viol, max_abs_err, fixed_viol, checked]
                    if t[3] == 0 or t[7] != t[3] or t[4] or t[5] or t[6]:
                        fails.append(f"{tag} {arm} {mode}: budget counters {t}")
                for m, sw in v[mode]["swaps"].items():
                    logged = sum(len(e["heads"]) for e in (v[mode]["swap_log"] or []))
                    if v[mode]["swap_log"] is not None and logged != sw[1]:
                        fails.append(f"{tag} {arm} {mode}: swap log {logged} != counter {sw[1]}")
    print(f"prompts {len(rows)}; failures {len(fails)}")
    for f in fails[:30]:
        print("  -", f)
    print("ACCEPTED" if not fails else "NOT ACCEPTED")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else
         __file__.rsplit("/src/", 1)[0] + "/results/mech_prompts_check.json")
