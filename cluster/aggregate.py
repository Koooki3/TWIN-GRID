"""Cluster aggregation for one cell (model, length, budget) and one arm pair (plan §3.4) — v3 after Codex reviews A1/A5
(batch 1) and Q2/Q3 (revision review). Nothing is computed before the gate passes (exit 2 on any failure).

Gate:
  - RULER: checkout HEAD == plan ruler_commit, and the sha256 of every RULER file the pipeline reads equals
    plan pinned_ruler_sources (so uncommitted edits of metric/yaml/template/constants are refused); prep manifest
    ruler_commit == plan;
  - data: n > 0; per task the frozen file exists, its sha256 equals the prep manifest's, its uid list is re-derived as
    "<sha16>:<task>:<i>" for every line and must equal the manifest's list exactly, and it has >= n lines;
  - prediction manifests (per arm, per task): all plan.protocol.required_code_keys present (missing is a failure, two
    arms missing the same key do not "agree"), model:revision recorded; data hash == the frozen file's; required
    fields present: ruler_commit == plan, tokens_to_generate == the official constant for the task, special_tokens
    True, stop_ids a non-empty list; argv: --budget, --block (arm's), --limit (= n), method listed, protocol values
    (--local, --rank, --prefill-chunk, --stop_words; absent = documented default), required switches present and
    forbidden switches absent; everything except the method list, block and save dir must be identical across arms
    (code/model identity, stop_ids, tokens_to_generate, special_tokens);
  - rows: no duplicate uids; every uid is a frozen line; lines 0..n-1 all present (rows for other frozen lines are
    ignored and counted); outputs == frozen answers; input_sha == sha256(input + answer_prefix)[:16] of the frozen
    line; token_sha identical between the two arms for every uid; sparse rows: budget counters present as integers
    (bool rejected): calls > 0, budget_checked_calls == calls, budget_violations == 0, max_abs_budget_error == 0,
    fixed_region_violations == 0 (missing keys fail, never count as 0).
Scores / roles / statistics: as v2 (unrounded official formulas vs official rounded task score within 0.005; PRIMARY /
SECONDARY / DESCRIPTIVE; cross-block pairs DESCRIPTIVE; stratified sign-flip permutation and bootstrap).
--json output carries the evidence identity needed by cluster/holm.py: plan_sha256, cell, n, pair, per-task data sha256,
the shared code/model identity, the protocol summary and a digest of every prediction manifest used.
v4 (plan v1.4): --units FILE binds the frozen unit-length file (cluster/unit_lengths.py; same data sha256, tokenizer
and block/local as the predictions) and prints the P-cov group means; --subset coverable restricts the estimand and the
tests to the coverable tasks (task groups re-derived from the file's per-sample lengths of lines 0..n-1) — the
registered Holm member ["cobs+g2f", "cobs", "coverable"].
Usage: aggregate.py --ruler ROOT --out-root DIR --model M --length L --budget B --n N --arm-a A --arm-b B
       --run-tag TAG [--units FILE [--subset coverable]] [--json OUT.json]"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_PATH = os.path.join(HERE, "plan_v1.json")
PLAN = json.load(open(PLAN_PATH))
PROTO = PLAN["protocol"]


def postprocess_pred(predict_str):            # verbatim from RULER scripts/eval/evaluate.py
    predict_str = predict_str.strip()
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()
    return predict_str


def unrounded(kind_fn_name, pred, refs):
    hits = [1.0 if r.lower() in pred.lower() else 0.0 for r in refs]
    return 100.0 * (sum(hits) / len(hits) if kind_fn_name == "string_match_all" else max(hits))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def counters_ok(t):
    keys = ("calls", "budget_checked_calls", "budget_violations", "max_abs_budget_error", "fixed_region_violations")
    if not isinstance(t, dict) or any(k not in t or not is_int(t[k]) for k in keys):
        return False
    return t["calls"] > 0 and t["budget_checked_calls"] == t["calls"] and t["budget_violations"] == 0 \
        and t["max_abs_budget_error"] == 0 and t["fixed_region_violations"] == 0


def manifest_problems(man, task, arm, n, budget, frozen_sha, tokgen):
    bad = []
    code = man.get("code") or {}
    miss = [k for k in PROTO["required_code_keys"] if k not in code]
    if miss:
        bad.append(f"missing code keys {miss}")
    if PROTO["revision_required"] and code.get("model:revision") in (None, "", "unrecorded"):
        bad.append("model revision not recorded")
    if code.get("data") != frozen_sha[:16]:
        bad.append("data hash differs from the frozen file")
    for f in PROTO["required_manifest_fields"]:
        if f not in man:
            bad.append(f"missing manifest field {f}")
    if man.get("ruler_commit") != PLAN["ruler_commit"]:
        bad.append("ruler_commit differs from the plan")
    if man.get("tokens_to_generate") != tokgen:
        bad.append(f"tokens_to_generate {man.get('tokens_to_generate')} != official {tokgen}")
    if man.get("special_tokens") is not True:
        bad.append("special_tokens is not True")
    if not isinstance(man.get("stop_ids"), list) or not man.get("stop_ids"):
        bad.append("stop_ids missing or empty")
    argv = man.get("argv") or []

    def val(k):
        return argv[argv.index(k) + 1] if k in argv and argv.index(k) + 1 < len(argv) else None
    if val("--budget") != str(budget) or val("--block") != str(arm["block"]) or val("--limit") != str(n) \
            or arm["method"] not in (val("--methods") or "").split(","):
        bad.append("argv budget/block/limit/method differ")
    for k, v in PROTO["argv_values"].items():
        if k in argv and val(k) != v:
            bad.append(f"argv {k}={val(k)} (protocol {v!r})")
    bad += [f"argv missing {s}" for s in PROTO["argv_switches_required"] if s not in argv]
    bad += [f"argv has forbidden {s}" for s in PROTO["argv_switches_forbidden"] if s in argv]
    return bad


def unit_groups(path, n, block, data_fp, identity):
    """Task groups from the frozen unit-length file, re-derived from its per-sample lengths of frozen lines 0..n-1
    (plan coverage_rule); the file must be bound to the same data, tokenizer, block and local."""
    rule, bad = PLAN["coverage_rule"], []
    u = json.load(open(path))
    if u.get("version") != rule["version"]:
        bad.append(f"unit file version {u.get('version')} != {rule['version']}")
    if u.get("block") != block or str(u.get("local")) != PROTO["argv_values"]["--local"] \
            or str(u.get("unit")) != PROTO["argv_values"]["--layout-unit"]:
        bad.append("unit file block/local/layout unit differ from the compared arms / protocol")
    tok = (identity.get("code") or {}).get("model:tokenizer.json")
    if not tok or not str(u.get("tokenizer_json_sha256", "")).startswith(tok):
        bad.append("unit file tokenizer differs from the prediction manifests' model:tokenizer.json")
    groups = {}
    for t in PLAN["tasks"]:
        r = (u.get("tasks") or {}).get(t)
        if not r or r.get("data_sha256") != data_fp.get(t) or len(r.get("samples", [])) < n:
            bad.append(f"{t}: unit file missing, bound to other data, or shorter than n")
            continue
        lens = [x for smp in r["samples"][:n] for x in smp.get("lens", [])]
        groups[t] = "no_unit" if not lens else ("coverable" if max(lens) <= block // 2 + 1 else "not_coverable")
    return groups, hashlib.sha256(open(path, "rb").read()).hexdigest(), bad


def main():
    ap = argparse.ArgumentParser()
    for k in ("ruler", "out-root", "model", "arm-a", "arm-b", "run-tag"):
        ap.add_argument(f"--{k}", required=True)
    for k in ("length", "budget", "n"):
        ap.add_argument(f"--{k}", type=int, required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--units", default="", help="frozen unit-length file (cluster/unit_lengths.py; plan coverage_rule)")
    ap.add_argument("--subset", default="", choices=["", "coverable"], help="registered task subset (needs --units)")
    a = ap.parse_args()
    bad = []
    head = subprocess.run(["git", "-C", a.ruler, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != PLAN["ruler_commit"]:
        bad.append(f"RULER checkout {head or '?'} != plan {PLAN['ruler_commit']}")
    for rel, h in PLAN["pinned_ruler_sources"].items():
        p = os.path.join(a.ruler, rel)
        if not os.path.exists(p) or sha(p) != h:
            bad.append(f"RULER source {rel} missing or differs from the pinned sha256")
    if bad:
        print("GATE FAILED:", *bad, sep="\n  ")
        sys.exit(2)
    metrics = load_mod(os.path.join(a.ruler, "scripts", "eval", "synthetic", "constants.py"), "ruler_eval_constants")
    tconst = load_mod(os.path.join(a.ruler, "scripts", "data", "synthetic", "constants.py"), "ruler_task_constants")
    import yaml
    cfg = yaml.safe_load(open(os.path.join(a.ruler, "scripts", "synthetic.yaml")))
    if a.n <= 0:
        bad.append("n must be positive")
    arms = {x: PLAN["arms"][x] for x in (a.arm_a, a.arm_b)}
    data_dir = os.path.join(a.out_root, "data", a.model, str(a.length))
    prep = json.load(open(os.path.join(data_dir, "prep_manifest.json")))
    if prep.get("ruler_commit") != PLAN["ruler_commit"]:
        bad.append("prep manifest ruler_commit differs from the plan")
    shared, digests, data_fp, diffs, table, ignored = {}, [], {}, {}, [], 0
    for task in PLAN["tasks"]:
        f = os.path.join(data_dir, task, "validation.jsonl")
        o = prep.get("outputs", {}).get(task)
        if not o or not os.path.exists(f) or sha(f) != o["sha256"]:
            bad.append(f"{task}: frozen data missing or changed")
            continue
        lines = [json.loads(l) for l in open(f)]
        derived = [f"{o['sha256'][:16]}:{task}:{i}" for i in range(len(lines))]
        if o.get("sample_uids") != derived:
            bad.append(f"{task}: prep uid list is not the re-derived full list for the frozen file")
            continue
        if len(lines) < a.n:
            bad.append(f"{task}: frozen file has {len(lines)} < n={a.n} lines")
            continue
        data_fp[task] = o["sha256"]
        want = derived[:a.n]
        kind = cfg[task]["task"]
        fn, tokgen = metrics.TASKS[kind]["metric_fn"], tconst.TASKS[kind]["tokens_to_generate"]
        per, tok_by_arm = {}, {}
        for name, arm in arms.items():
            d = os.path.join(a.out_root, "pred", a.model, str(a.length), f"budget{a.budget}", f"b{arm['block']}",
                             a.run_tag)
            man_p = os.path.join(d, f"manifest.{task}.json")
            p = os.path.join(d, f"{arm['method']}@{a.budget}", f"{task}.jsonl")
            if not os.path.exists(man_p) or not os.path.exists(p):
                bad.append(f"{task} {name}: missing manifest or predictions in {d}")
                continue
            man = json.load(open(man_p))
            bad += [f"{task} {name}: {x}" for x in manifest_problems(man, task, arm, a.n, a.budget, o["sha256"], tokgen)]
            ident = dict(code={k: v for k, v in (man.get("code") or {}).items() if k != "data"},
                         stop_ids=man.get("stop_ids"), tokens_to_generate=man.get("tokens_to_generate"),
                         special_tokens=man.get("special_tokens"))
            key = json.dumps(ident, sort_keys=True)
            if shared.setdefault(task, key) != key:
                bad.append(f"{task}: code/model/protocol identity differs between arms")
            digests.append(sha(man_p))
            rows = [json.loads(l) for l in open(p)]
            uids = [r.get("sample_uid") for r in rows]
            if len(set(uids)) != len(uids) or not set(uids) <= set(derived) or not set(want) <= set(uids):
                bad.append(f"{task} {name}: duplicate, foreign or missing sample uids")
                continue
            ignored += sum(1 for u in uids if u not in set(want))
            by = {r["sample_uid"]: r for r in rows}
            sc, toks = [], {}
            for i, u in enumerate(want):
                r, ln = by[u], lines[i]
                if r.get("outputs") != ln["outputs"] or r.get("input_sha") != hashlib.sha256(
                        (ln["input"] + ln.get("answer_prefix", "")).encode()).hexdigest()[:16]:
                    bad.append(f"{task} {name} {u}: answers/input differ from the frozen line")
                    break
                if arm["method"] != "dense" and not counters_ok(r.get("tokens")):
                    bad.append(f"{task} {name} {u}: budget counters missing, non-integer or non-zero violations")
                    break
                toks[u] = r.get("token_sha")
                sc.append((postprocess_pred(r["pred"]), ln["outputs"]))
            tok_by_arm[name] = toks
            if len(sc) == a.n:
                un = np.array([unrounded(fn.__name__, pr, rf) for pr, rf in sc])
                off = fn([pr for pr, _ in sc], [rf for _, rf in sc])
                if abs(un.mean() - off) > 0.005 + 1e-9:
                    bad.append(f"{task} {name}: unrounded mean {un.mean():.4f} vs official {off} beyond rounding")
                per[name] = (un, off)
        if len(tok_by_arm) == 2 and any(tok_by_arm[a.arm_a].get(u) is None or
                                        tok_by_arm[a.arm_a].get(u) != tok_by_arm[a.arm_b].get(u) for u in want):
            bad.append(f"{task}: token_sha missing or differs between arms")
        if len(per) == 2:
            diffs[task] = per[a.arm_a][0] - per[a.arm_b][0]
            table.append((task, per[a.arm_a][1], per[a.arm_b][1], diffs[task].mean()))
    cross_task = {json.dumps({k: v for k, v in json.loads(x).items() if k != "tokens_to_generate"}, sort_keys=True)
                  for x in shared.values()}           # tokens_to_generate is per task (checked against the official value)
    if len(cross_task) > 1:
        bad.append("code/model/protocol identity differs across tasks")
    if bad or len(diffs) != len(PLAN["tasks"]):
        print("GATE FAILED:", *bad[:20], sep="\n  ")
        sys.exit(2)
    pc = PLAN["primary_cell"]
    same_block = arms[a.arm_a]["block"] == arms[a.arm_b]["block"]
    in_primary_cell = (a.model, a.length, a.budget) == (pc["model"], pc["length"], pc["budget"]) and a.n == pc["samples"]
    groups, units_sha = {}, None
    if a.subset and not a.units:
        print("GATE FAILED:\n  --subset needs --units (the frozen unit-length file)")
        sys.exit(2)
    if a.units:
        groups, units_sha, ubad = unit_groups(a.units, a.n, arms[a.arm_a]["block"], data_fp,
                                              json.loads(next(iter(cross_task))))
        if not same_block:
            ubad.append("unit groups are defined per block size; the two arms differ")
        if ubad:
            print("GATE FAILED:", *ubad, sep="\n  ")
            sys.exit(2)
    pair = [a.arm_a, a.arm_b] + ([a.subset] if a.subset else [])
    role = "DESCRIPTIVE"
    if in_primary_cell and same_block and pair == PLAN["primary_pair"]:
        role = "PRIMARY"
    elif in_primary_cell and same_block and pair in PLAN["secondary_family"]:
        role = "SECONDARY"
    all_diffs = diffs
    if a.subset:
        diffs = {t: d for t, d in diffs.items() if groups[t] == a.subset}
        if not diffs:
            print(f"GATE FAILED:\n  subset {a.subset} is empty")
            sys.exit(2)
    D = float(np.mean([d.mean() for d in diffs.values()]))
    rng = np.random.default_rng(0)
    boot = np.zeros(10_000)
    for d in diffs.values():
        boot += d[rng.integers(0, d.size, size=(10_000, d.size))].mean(1)
    boot /= len(diffs)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    p = None
    if role != "DESCRIPTIVE":
        R = 100_000
        perm = np.zeros(R)
        for d in diffs.values():
            perm += (rng.choice([-1.0, 1.0], size=(R, d.size)) * d).mean(1)
        perm /= len(diffs)
        p = float((1 + np.sum(np.abs(perm) >= abs(D) - 1e-12)) / (1 + R))
    print(f"[{role}] cell {a.model} L={a.length} budget={a.budget} n={a.n}/task; {a.arm_a} vs {a.arm_b}"
          f"{'' if same_block else ' (different block sizes: descriptive only)'}; rows beyond n ignored: {ignored}")
    for t, oa, ob, md in table:
        print(f"  {t:16s} official {a.arm_a} {oa:6.2f}  {a.arm_b} {ob:6.2f}  per-sample mean diff {md:+6.2f}")
    if groups:
        for g in ("coverable", "not_coverable", "no_unit"):
            ts = [t for t in all_diffs if groups[t] == g]
            if ts:
                print(f"  group {g:13s} ({len(ts)} tasks): mean per-task diff "
                      f"{np.mean([all_diffs[t].mean() for t in ts]):+.3f} (P-cov, descriptive)")
    if a.subset:
        print(f"subset {a.subset}: {', '.join(diffs)} (from unit file {units_sha[:16]})")
    print(f"macro difference D = {D:+.3f} points; stratified bootstrap 95% [{lo:+.3f}, {hi:+.3f}]"
          + (f"; stratified sign-flip permutation p = {p:.6f} ({'unadjusted; Holm via cluster/holm.py' if role == 'SECONDARY' else 'primary test'})"
             if p is not None else "; no test (descriptive)"))
    if a.json:
        json.dump(dict(role=role, plan_sha256=sha(PLAN_PATH), model=a.model, length=a.length, budget=a.budget, n=a.n,
                       pair=pair, D=D, ci95=[lo, hi], p=p, data_sha256=data_fp,
                       identity=json.loads(next(iter(cross_task))),
                       manifest_digest=hashlib.sha256("".join(sorted(digests)).encode()).hexdigest(),
                       per_task=[dict(task=t, official_a=oa, official_b=ob, diff=md) for t, oa, ob, md in table],
                       subset=a.subset or None, subset_tasks=sorted(diffs) if a.subset else None,
                       unit_groups=groups or None, units_sha256=units_sha),
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
