"""Run descriptor for plan v2 (Codex implementation review I1–I3): one structured, hashed record that binds a v2 run to
its plan, model weights, data, unit file, code, adapter acceptance and memory probe, written BEFORE any prediction,
re-verified immediately before predicting, and bound to every prediction file afterwards. The frozen src/ruler_pred.py
is not changed; binding is done with sidecar files.

  register  --tier T --model M --models-dir DIR --hf-repo REPO --revision SHA40 --out-root DIR --units FILE
            --adapter-verdict FILE --adapter-report FILE --adapter-pred-root DIR [--adapter-tie-diag FILE ...]
            --probe FILE [--probe ...] --out RD.json
     Refuses unless: the snapshot has config/tokenizer/tokenizer_config/generation_config and either model.safetensors
     or model.safetensors.index.json with EVERY shard of its weight_map present; the revision is 40 hex;
     the adapter decision RE-DERIVED from the structured report (acceptance_problems; Codex I2-a) is ACCEPTED: a JSON
     report with empty input_problems, a same-input step for every plan task with >= 5 rows, order / cache-isolation
     steps for both same-family pairs (cobs/cobs+g2f, quest/quest+g2f) identical and covered in ab/ba/single, no
     mismatch (strict, passed true) or every mismatch with a tie diagnostic (tie true, equal official scores, <= 1 per
     task, <= floor(0.02 x rows)); the verdict text equals the re-derived decision; every acceptance manifest under
     --adapter-pred-root carries this model's and code's ids (current accept_ties alone would accept empty steps);
     every probe (probe_problems; Codex I2-b) is for this model directory / length / budget, has prefill and decode:<m>
     for every selected method of its block, finite non-negative phase values, positive capacity, top-level peak equal
     to the max over phases, peak x 1.1 <= capacity, and a receipt (probe-receipt, written right after the probe)
     binding it to the current weights (full sha256, size, mtime), code and parameters; probes cover every block of the tier; the
     unit file and prep manifest exist. Records full sha256 (and size, mtime) of every weight file, sha256 of
     plan_v2.json, the prep manifest and per-task data, the unit file, the code (src/ruler_pred.py, src/sparse_attn.py,
     src/v2_stats.py, cluster/aggregate.py, cluster/aggregate_v2.py, this file), the re-derived adapter decision with
     the report / verdict / tie-diagnostic hashes and acceptance code hashes, the probes and their receipts, the
     tier's allowed prediction matrix and the time.
  probe-receipt --probe P --tier T --model M --models-dir DIR   (prep, right after each probe run)
  verify    --rd RD.json --models-dir DIR --out-root DIR
     Recomputes every recorded hash (weights included) and the plan/code hashes; exit 1 on any difference. The
     generated predict phase runs this before its first prediction.
  bind      --rd RD.json --save-dir D --task T --methods m1,m2 --budget B
     After a prediction job: writes D/binding.<task>.json = {rd_sha256, manifest sha256, sha256 of every method's
     prediction file}; refuses if the manifest's model/code ids differ from the descriptor.
check(rd_path) is used by cluster/aggregate_v2.py: schema validation, current plan/code hashes equal the recorded ones.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLAN2_PATH = os.path.join(HERE, "plan_v2.json")
HEX64, HEX40 = re.compile(r"^[0-9a-f]{64}$"), re.compile(r"^[0-9a-f]{40}$")
META = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
CODE = ("src/ruler_pred.py", "src/sparse_attn.py", "src/v2_stats.py", "cluster/aggregate.py", "cluster/aggregate_v2.py",
        "cluster/v2_run_descriptor.py")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def weight_files(mdir):
    """The complete weight file set of a snapshot (index-declared shards), or raise."""
    for f in META:
        if not os.path.exists(os.path.join(mdir, f)):
            raise ValueError(f"{mdir}: missing {f}")
    single, index = os.path.join(mdir, "model.safetensors"), os.path.join(mdir, "model.safetensors.index.json")
    if os.path.exists(single):                      # same precedence as src/ruler_pred.py model_identity
        return list(META) + ["model.safetensors"]
    if os.path.exists(index):
        shards = sorted(set(json.load(open(index))["weight_map"].values()))
        missing = [s for s in shards if not os.path.exists(os.path.join(mdir, s))]
        if not shards or missing:
            raise ValueError(f"{mdir}: index declares shards {missing or 'none'} that are missing")
        return list(META) + ["model.safetensors.index.json"] + shards
    raise ValueError(f"{mdir}: neither model.safetensors nor model.safetensors.index.json")


def file_record(mdir, f):
    p = os.path.join(mdir, f)
    st = os.stat(p)
    return dict(sha256=sha(p), size=st.st_size, mtime=int(st.st_mtime))


def code_hashes():
    return {c: sha(os.path.join(ROOT, c)) for c in CODE}


ACCEPT_CODE = ("cluster/accept_adapter.py", "cluster/accept_tie_diag.py", "cluster/accept_ties.py")
PROBE_CODE = ("cluster/probe_memory.py", "src/sparse_attn.py")
ORDER_PAIRS = (("cobs", "cobs+g2f"), ("quest", "quest+g2f"))      # same-family pairs that share cache keys
MIN_SAME_INPUT_ROWS = 5                                           # guide §3: 13 tasks x 5 fixed samples


def ulp_bf16(x):
    """plan v1.4 §3.2: ulp_bf16(x) = 2^(floor(log2 |x|) - 7); 2^-133 at x = 0, exactly as the frozen producer
    cluster/accept_tie_diag.py (Codex I2-a.2 compatibility)."""
    import math
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x else 2.0 ** -133


def tie_problems(d):
    """Full schema of a cluster/accept_tie_diag.py diagnostic and the tie predicate RE-COMPUTED from its fields
    (Codex I2-a.2): input_equal and swap and margin <= 2 ulp, margin = l1 - l2, ulp = ulp_bf16(max |l|)."""
    import math
    bad = []
    if not isinstance(d, dict):
        return ["not an object"]
    for k, typ in (("uid", str), ("input_equal", bool), ("swap", bool), ("tie", bool)):
        if not isinstance(d.get(k), typ):
            bad.append(f"{k} missing or not {typ.__name__}")
    ds = d.get("divergence_step")
    if not (isinstance(ds, int) and not isinstance(ds, bool) and ds >= 0):
        bad.append("divergence_step missing or not a non-negative int")
    lg = d.get("ref_logits")
    fin = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)  # noqa: E731
    if not (isinstance(lg, list) and len(lg) == 2 and all(fin(x) for x in lg) and lg[0] >= lg[1]):
        bad.append("ref_logits missing, non-finite or not ordered top-2")
    elif not (fin(d.get("margin")) and fin(d.get("ulp")) and d["ulp"] > 0):
        bad.append("margin / ulp missing or non-finite")
    else:
        if abs(d["margin"] - (lg[0] - lg[1])) > 1e-6:
            bad.append("margin != l1 - l2")
        m = max(abs(lg[0]), abs(lg[1]))
        if d["ulp"] != ulp_bf16(m) and abs(d["ulp"] - ulp_bf16(m)) > 1e-12 * ulp_bf16(m):
            bad.append("ulp != ulp_bf16(max |l|)")
    if not bad:
        tie = d["input_equal"] and d["swap"] and d["margin"] <= 2 * d["ulp"]
        if tie != d["tie"]:
            bad.append("tie flag contradicts input_equal / swap / margin <= 2 ulp")
        elif not tie:
            bad.append("not a tie (input_equal, swap and margin <= 2 ulp all required)")
    return bad


def acceptance_problems(report_p, verdict_p, diag_ps, pred_root, mdir, plan):
    """Re-derive the adapter decision from the structured evidence (Codex I2-a); never trust the verdict text alone.
    Returns (decision or None, problems). decision in {"ACCEPTED (strict)", "ACCEPTED_WITH_TIES"}."""
    bad = []
    try:
        rep = json.load(open(report_p))
        assert isinstance(rep, dict)
    except (OSError, ValueError, AssertionError):
        return None, ["adapter report is not a JSON object"], None, set()
    if rep.get("input_problems") != []:
        bad.append("adapter report has input problems (or the field is missing)")
    steps = rep.get("steps")
    if not isinstance(steps, dict) or not steps:
        return None, bad + ["adapter report has no steps"], None, set()
    rows = []
    for t in plan["tasks"]:
        s = steps.get(f"same_input:{t}")
        if not isinstance(s, dict) or not isinstance(s.get("rows"), list) or len(s["rows"]) < MIN_SAME_INPUT_ROWS \
                or s.get("n") != len(s["rows"]):
            bad.append(f"adapter same-input step for {t} missing or with < {MIN_SAME_INPUT_ROWS} rows")
            continue
        for r in s["rows"]:
            if not isinstance(r, dict) or not isinstance(r.get("uid"), str) or "same_text" not in r:
                bad.append(f"adapter same-input row malformed in {t}")
                break
            rows.append((t, r))
    for a, b in ORDER_PAIRS:
        for m in (a, b):
            hits = [v for k, v in steps.items() if k.startswith("order:") and k.endswith(f":{m}") and a in k and b in k]
            if not hits:
                bad.append(f"adapter order / cache-isolation step for pair {a}/{b} method {m} missing")
            for v in hits:
                cov = (v or {}).get("covered_rows") or {}
                if v.get("status") != "identical" or set(cov) != {"ab", "ba", "single"} or min(cov.values()) <= 0:
                    bad.append(f"adapter order step {a}/{b}:{m} not identical or not covered")
    mism = [(t, r) for t, r in rows if not r["same_text"] or r.get("score_ref") != r.get("score_ours")]
    decision = None
    if not bad:
        if not mism:
            decision = "ACCEPTED (strict)"
            if rep.get("passed") is not True:
                bad.append("adapter report: no mismatch but passed is not true")
        else:
            diags = {}
            for p in diag_ps:
                try:
                    d = json.load(open(p))
                    diags[d["uid"]] = d
                except (OSError, ValueError, KeyError, TypeError):
                    bad.append(f"tie diagnostic {p} unreadable")
            per = {}
            for t, r in mism:
                d = diags.get(r["uid"])
                tb = tie_problems(d) if d is not None else ["missing"]
                if d is None or d.get("tie") is not True or tb:
                    bad.append(f"mismatch {r['uid']}: no tie diagnostic with tie = true" + (f" ({'; '.join(tb)})" if tb else ""))
                if r.get("score_ref") != r.get("score_ours"):
                    bad.append(f"mismatch {r['uid']}: official scores differ")
                per[t] = per.get(t, 0) + 1
            bad += [f"{t}: {c} ties (> 1 per task)" for t, c in per.items() if c > 1]
            if len(mism) > len(rows) * 2 // 100:
                bad.append(f"{len(mism)} ties > floor(0.02 * {len(rows)})")
            if set(diags) - {r["uid"] for _, r in mism}:
                bad.append("tie diagnostics for rows that are not mismatches")
            decision = "ACCEPTED_WITH_TIES"
    try:
        verdict = open(verdict_p).read().strip().splitlines()[0]
    except (OSError, IndexError):
        verdict = ""
    if decision and not verdict.startswith(decision):
        bad.append(f"verdict '{verdict[:40]}' does not match the decision re-derived from the report ({decision})")
    return (decision if not bad else None), bad, rep, {r["uid"] for _, r in mism}


def acceptance_inputs(pred_root, data_dir, plan, order_task, rep, mdir):
    """Link the report to the complete acceptance inputs (Codex I2-a.1). Layout of cluster/accept_adapter.py:
    dense/ (every plan task), order_ab_<A>__<B>/, order_ba_<A>__<B>/, single_<A>/, single_<B>/ (order task) for both
    same-family pairs. Every run: manifest.<task>.json with this model's and code's ids and the frozen data id, the
    method prediction files present, uids frozen and unique; the report's same-input uids per task equal the dense
    run's, its order steps count the order task's uids and the ab / ba / single runs share them.
    Returns ({relative path: sha256} of every manifest and prediction file, problems)."""
    bad, inputs = [], {}
    try:
        files = weight_files(mdir)
    except ValueError as e:
        return {}, [str(e)]
    want = {"sparse_attn.py": sha(os.path.join(ROOT, "src/sparse_attn.py"))[:16],
            "ruler_pred.py": sha(os.path.join(ROOT, "src/ruler_pred.py"))[:16]}
    for f in files:
        rec = file_record(mdir, f)
        want[f"model:{f}"] = f"size={rec['size']} mtime={rec['mtime']}" if f.endswith(".safetensors") else rec["sha256"][:16]
    try:
        prep = json.load(open(os.path.join(data_dir, "prep_manifest.json")))["outputs"]
    except (OSError, ValueError, KeyError):
        return {}, ["acceptance data: prep manifest missing"]
    budgets = set()

    def run(sub, task, methods):
        d = os.path.join(pred_root, sub)
        mp = os.path.join(d, f"manifest.{task}.json")
        try:
            man = json.load(open(mp))
        except (OSError, ValueError):
            bad.append(f"acceptance run {sub}/{task}: manifest missing")
            return None
        code = man.get("code") or {}
        diff = [k for k, v in want.items() if code.get(k) != v]
        extra = [k for k in code if k.startswith("model:") and k != "model:revision" and k not in want]
        if diff or extra:
            bad.append(f"acceptance manifest {sub}/{task} is for another model/code ({(diff + extra)[:3]})")
        if task not in prep or code.get("data") != prep[task]["sha256"][:16]:
            bad.append(f"acceptance manifest {sub}/{task}: data id differs from the frozen {task} data")
            return None
        argv = man.get("argv") or []
        bud = argv[argv.index("--budget") + 1] if "--budget" in argv and argv.index("--budget") + 1 < len(argv) else None
        budgets.add(bud)
        inputs[os.path.relpath(mp, pred_root)] = sha(mp)
        out = {}
        for m in methods:
            p = os.path.join(d, f"{m}@{bud}", f"{task}.jsonl")
            if not os.path.exists(p):
                bad.append(f"acceptance run {sub}/{task}: prediction file for {m} missing")
                return None
            try:
                uids = [json.loads(x).get("sample_uid") for x in open(p)]
            except (ValueError, AttributeError):
                bad.append(f"acceptance run {sub}/{task} {m}: prediction file is not JSON lines")
                return None
            if not uids or len(set(uids)) != len(uids) or not set(uids) <= set(prep[task]["sample_uids"]):
                bad.append(f"acceptance run {sub}/{task} {m}: empty, duplicate or foreign uids")
                return None
            inputs[os.path.relpath(p, pred_root)] = sha(p)
            out[m] = set(uids)
        return out
    steps = rep.get("steps") or {}
    for t in plan["tasks"]:
        u = run("dense", t, ["dense"])
        rep_uids = {r.get("uid") for r in (steps.get(f"same_input:{t}") or {}).get("rows", []) if isinstance(r, dict)}
        if u is not None and u["dense"] != rep_uids:
            bad.append(f"acceptance report same-input uids for {t} differ from the dense acceptance run")
    for a, b in ORDER_PAIRS:
        sets = []
        for sub, ms in ((f"order_ab_{a}__{b}", [a, b]), (f"order_ba_{a}__{b}", [b, a]), (f"single_{a}", [a]), (f"single_{b}", [b])):
            u = run(sub, order_task, ms)
            if u is not None:
                sets += list(u.values())
        if sets and any(s != sets[0] for s in sets):
            bad.append(f"acceptance order runs for {a}/{b} do not share one uid set")
        for m in (a, b):
            for k, v in steps.items():
                if k.startswith("order:") and k.endswith(f":{m}") and a in k and b in k and sets and v.get("n") != len(sets[0]):
                    bad.append(f"acceptance report order step {k} counts {v.get('n')} rows, runs have {len(sets[0])}")
    if len(budgets) != 1:
        bad.append(f"acceptance runs use budgets {sorted(map(str, budgets))}, expected one")
    return inputs, bad


PROV_SUFFIX = ".provenance.json"


def snapshot(paths):
    """{path/relative file: sha256} for every file under the given directories / files (provenance files excluded)."""
    out = {}
    for p in paths:
        if os.path.isdir(p):
            for dp, _, fs in os.walk(p):
                for f in fs:
                    if not f.endswith(PROV_SUFFIX):
                        fp = os.path.join(dp, f)
                        out[os.path.realpath(fp)] = sha(fp)
        elif os.path.exists(p):
            out[os.path.realpath(p)] = sha(p)
        else:
            out[os.path.realpath(p)] = None
    return out


def eval_run(a):
    """Evaluation-boundary provenance (Codex I2-a.1): hash the declared inputs, the full model and the code, run the
    evaluator (accept_adapter.py / accept_tie_diag.py / accept_ties.py, unchanged), require the inputs unchanged and the
    output newly created by this run, then write <out>.provenance.json. Refuses if <out> already exists (a new
    evaluation needs a new attempt path). Exit code = the evaluator's."""
    import subprocess
    mdir = os.path.join(a.models_dir, a.model)
    if os.path.exists(a.out) or os.path.exists(a.out + PROV_SUFFIX):
        print(f"EVAL-RUN REFUSED: {a.out} already exists (start a new attempt)")
        sys.exit(1)
    pre, model, code = snapshot(a.watch), {f: file_record(mdir, f) for f in weight_files(mdir)}, \
        {c: sha(os.path.join(ROOT, c)) for c in ACCEPT_CODE + ("src/ruler_pred.py", "src/sparse_attn.py")}
    t0 = time.time()
    cmd = a.evaluator[1:] if a.evaluator and a.evaluator[0] == "--" else a.evaluator
    res = subprocess.run(cmd, stdout=subprocess.PIPE if a.stdout_to_out else None, text=True)
    if a.stdout_to_out:
        open(a.out, "w").write(res.stdout)
    bad = []
    if snapshot(a.watch) != pre:
        bad.append("inputs changed during the evaluation")
    if {f: file_record(mdir, f) for f in weight_files(mdir)} != model or \
            {c: sha(os.path.join(ROOT, c)) for c in code} != code:
        bad.append("model or code changed during the evaluation")
    if not os.path.exists(a.out) or os.stat(a.out).st_mtime < t0 - 1:
        bad.append("the evaluator did not create the output in this run")
    if bad:
        print("EVAL-RUN REFUSED:", *bad, sep="\n  ")
        sys.exit(1)
    json.dump(dict(cmd=cmd, rc=res.returncode, started=t0, ended=time.time(), inputs=pre, model=model, code=code,
                   output=os.path.realpath(a.out), output_sha256=sha(a.out)),
              open(a.out + PROV_SUFFIX, "w"), indent=1, sort_keys=True)
    sys.exit(res.returncode)


def provenance_problems(out_p, inputs_now, model_now, code_now, need_rc0=False, required=()):
    """The output must carry an eval-run provenance whose output hash is the current file's, whose snapshot CONTAINS
    every required input of its role (derived independently from the acceptance layout, never from the provenance's
    own declaration; Codex I2-a.1) with the current hash, whose other recorded inputs equal the current ones, and
    whose model and code equal the current ones."""
    try:
        pv = json.load(open(out_p + PROV_SUFFIX))
    except (OSError, ValueError):
        return [f"{os.path.basename(out_p)}: evaluation provenance missing (not produced through eval-run)"]
    bad = []
    if pv.get("output_sha256") != sha(out_p):
        bad.append(f"{os.path.basename(out_p)} changed after its evaluation")
    rec = pv.get("inputs") or {}
    for p in sorted(required):
        if p not in rec:
            bad.append(f"{os.path.basename(out_p)}: evaluation did not cover required input {p}")
            break
        if not os.path.exists(p) or rec[p] != sha(p):
            bad.append(f"{os.path.basename(out_p)}: required input {p} differs from its evaluation snapshot")
            break
    for p, h in rec.items():
        if inputs_now.get(p, "absent") != h:
            bad.append(f"{os.path.basename(out_p)}: evaluation input {p} differs from the current acceptance inputs")
            break
    if pv.get("model") != model_now or pv.get("code") != code_now:
        bad.append(f"{os.path.basename(out_p)}: evaluated with another model or code")
    if need_rc0 and pv.get("rc") != 0:
        bad.append(f"{os.path.basename(out_p)}: evaluator exit code {pv.get('rc')}")
    return bad


def order_problems(pred_root, order_task, budget, rep):
    """Re-derive the order / cache-isolation evidence from the bound rows exactly as accept_adapter.py does (identical =
    ab == ba == single prediction for every uid; covered_rows from gen_in_candidates) and compare with the report."""
    bad = []
    steps = rep.get("steps") or {}

    def rows(sub, m):
        p = os.path.join(pred_root, sub, f"{m}@{budget}", f"{order_task}.jsonl")
        if not os.path.exists(p):
            return None
        try:
            return {json.loads(x)["sample_uid"]: json.loads(x) for x in open(p)}
        except (ValueError, KeyError, TypeError):
            return None
    for a, b in ORDER_PAIRS:
        for m in (a, b):
            runs = dict(ab=rows(f"order_ab_{a}__{b}", m), ba=rows(f"order_ba_{a}__{b}", m), single=rows(f"single_{m}", m))
            if any(v is None for v in runs.values()):
                bad.append(f"order runs for {a}/{b}:{m} incomplete")
                continue
            uids = sorted(runs["single"])
            identical = all(runs["ab"].get(u, {}).get("pred") == runs["ba"].get(u, {}).get("pred") ==
                            runs["single"][u].get("pred") for u in uids)
            cov = {k: sum(1 for u in uids if (runs[k].get(u) or {}).get("gen_in_candidates")) for k in runs}
            status = "identical" if identical else "DIFFERENT"
            if min(cov.values()) == 0 and identical:
                status = "not covered"
            keys = [k for k in steps if k.startswith("order:") and k.endswith(f":{m}") and a in k and b in k]
            for k in keys:
                v = steps[k]
                if v.get("status") != status or v.get("n") != len(uids) or v.get("covered_rows") != cov:
                    bad.append(f"report {k} (status {v.get('status')}, covered {v.get('covered_rows')}) contradicts the "
                               f"bound runs (status {status}, covered {cov})")
            if not identical:
                bad.append(f"order runs for {a}/{b}:{m}: predictions differ across ab / ba / single")
            elif min(cov.values()) == 0:
                bad.append(f"order runs for {a}/{b}:{m}: no row covered in some run")
    return bad


def accept_receipt(a):
    """Written right after the adapter acceptance (prep): binds the report, verdict and tie diagnostics to the complete
    acceptance inputs, the full model content hashes and the code (Codex I2-a.1). Refuses on any problem."""
    plan = json.load(open(PLAN2_PATH))
    mdir = os.path.join(a.models_dir, a.model)
    decision, bad, rep, _ = acceptance_problems(a.report, a.verdict, a.tie_diag, a.pred_root, mdir, plan)
    if a.order_task != plan["acceptance"]["order_task"]:
        bad.append(f"order task {a.order_task} != plan {plan['acceptance']['order_task']}")
    inputs, ibad = acceptance_inputs(a.pred_root, a.data_dir, plan, a.order_task, rep or {}, mdir) if rep else ({}, [])
    bad += ibad
    if rep is not None:
        bad += evidence_problems(a.pred_root, a.data_dir, a.order_task, rep, a.report, a.verdict, a.tie_diag, mdir)
    rpath = a.report + ".receipt.json"
    if os.path.exists(rpath):                    # receipts are never overwritten (Codex I2-a.1)
        bad.append(f"acceptance receipt {rpath} already exists: receipts are not rewritten; a new evaluation needs a "
                   "new attempt directory")
    if bad:
        print("ACCEPTANCE RECEIPT REFUSED:", *bad, sep="\n  ")
        sys.exit(1)
    json.dump(dict(decision=decision, order_task=a.order_task, data_dir=os.path.realpath(a.data_dir),
                   data={t: o["sha256"] for t, o in json.load(open(os.path.join(a.data_dir, "prep_manifest.json")))["outputs"].items()},
                   model={f: file_record(mdir, f) for f in weight_files(mdir)},
                   code={c: sha(os.path.join(ROOT, c)) for c in ACCEPT_CODE + ("src/ruler_pred.py", "src/sparse_attn.py")},
                   report_sha256=sha(a.report), verdict_sha256=sha(a.verdict),
                   tie_diagnostics={os.path.basename(p): sha(p) for p in a.tie_diag}, inputs=inputs,
                   provenance={os.path.basename(p): sha(p + PROV_SUFFIX) for p in [a.report, a.verdict] + list(a.tie_diag)}),
              open(rpath, "x"), indent=1, sort_keys=True)
    print(f"acceptance receipt {a.report}.receipt.json")


def evidence_problems(pred_root, data_dir, order_task, rep, report_p, verdict_p, diag_ps, mdir):
    """Order evidence re-derived from the rows, and evaluation-boundary provenance of the report, the verdict
    (exit 0 required) and every tie diagnostic against the current inputs, model and code."""
    bad = []
    try:
        man = json.load(open(os.path.join(pred_root, f"single_{ORDER_PAIRS[0][0]}", f"manifest.{order_task}.json")))
        argv = man.get("argv") or []
        budget = argv[argv.index("--budget") + 1]
    except (OSError, ValueError, IndexError):
        return ["acceptance order runs: budget unreadable"]
    bad += order_problems(pred_root, order_task, budget, rep)
    try:
        model_now = {f: file_record(mdir, f) for f in weight_files(mdir)}
    except ValueError as e:
        return bad + [str(e)]
    code_now = {c: sha(os.path.join(ROOT, c)) for c in ACCEPT_CODE + ("src/ruler_pred.py", "src/sparse_attn.py")}
    inputs_now = snapshot([pred_root, data_dir, report_p] + list(diag_ps))
    # required inputs per role, derived from the acceptance layout (not from what an eval-run chose to watch)
    plan = json.load(open(PLAN2_PATH))
    layout, lbad = acceptance_inputs(pred_root, data_dir, plan, order_task, rep, mdir)
    if lbad or not layout:
        bad.append("required evaluation inputs cannot be derived from the acceptance layout (refused)")
    data_req = {os.path.realpath(os.path.join(data_dir, "prep_manifest.json"))} | \
        {os.path.realpath(os.path.join(data_dir, t, "validation.jsonl")) for t in plan["tasks"]}
    run_req = {os.path.realpath(os.path.join(pred_root, rel)) for rel in layout} | data_req     # report, tie diagnostics
    verdict_req = {os.path.realpath(report_p)} | {os.path.realpath(p) for p in diag_ps}       # accept_ties reads these
    bad += provenance_problems(report_p, inputs_now, model_now, code_now, required=run_req)
    bad += provenance_problems(verdict_p, inputs_now, model_now, code_now, need_rc0=True, required=verdict_req)
    for p in diag_ps:
        bad += provenance_problems(p, inputs_now, model_now, code_now, need_rc0=True, required=run_req)
    return bad


def acceptance_receipt_problems(a, mdir, plan, prep_tasks):
    """Registration-time check of the acceptance receipt against the current files (full content hashes)."""
    rp = a.adapter_report + ".receipt.json"
    try:
        rc = json.load(open(rp))
    except (OSError, ValueError):
        return ["acceptance receipt missing or unreadable"]
    bad = []
    if rc.get("report_sha256") != sha(a.adapter_report) or rc.get("verdict_sha256") != sha(a.adapter_verdict) \
            or rc.get("tie_diagnostics") != {os.path.basename(p): sha(p) for p in a.adapter_tie_diag}:
        bad.append("acceptance receipt is for another report / verdict / tie diagnostics (stale)")
    try:
        now = {f: file_record(mdir, f) for f in weight_files(mdir)}
    except ValueError as e:
        return bad + [str(e)]
    if rc.get("model") != now:
        bad.append("acceptance receipt model content differs from the current weights (full sha256)")
    if rc.get("code") != {c: sha(os.path.join(ROOT, c)) for c in ACCEPT_CODE + ("src/ruler_pred.py", "src/sparse_attn.py")}:
        bad.append("acceptance receipt code differs from the current code")
    if rc.get("order_task") != plan["acceptance"]["order_task"]:
        bad.append(f"acceptance order task {rc.get('order_task')} != plan {plan['acceptance']['order_task']}")
    # the acceptance has its own small frozen data set (plan acceptance.rows_per_task per task): re-hash it now
    ad = a.adapter_data_dir
    try:
        cur = {t: o["sha256"] for t, o in json.load(open(os.path.join(ad, "prep_manifest.json")))["outputs"].items()}
        files_ok = all(os.path.exists(os.path.join(ad, t, "validation.jsonl")) and
                       sha(os.path.join(ad, t, "validation.jsonl")) == h for t, h in cur.items())
    except (OSError, ValueError, KeyError):
        cur, files_ok = None, False
    if rc.get("data") != cur or not files_ok or os.path.realpath(ad) != rc.get("data_dir") \
            or sorted(cur or {}) != sorted(plan["tasks"]):
        bad.append("acceptance receipt data differs from the acceptance data directory (or does not cover every task)")
    for rel, h in (rc.get("inputs") or {}).items():
        p = os.path.join(a.adapter_pred_root, rel)
        if not os.path.exists(p) or sha(p) != h:
            bad.append(f"acceptance input {rel} changed or missing since the receipt")
            break
    if not rc.get("inputs"):
        bad.append("acceptance receipt lists no inputs")
    for name, h in (rc.get("provenance") or {}).items():
        p = os.path.join(os.path.dirname(a.adapter_report), name) + PROV_SUFFIX
        if name == os.path.basename(a.adapter_verdict):
            p = a.adapter_verdict + PROV_SUFFIX
        if not os.path.exists(p) or sha(p) != h:
            bad.append(f"evaluation provenance of {name} changed or missing since the receipt")
    if not rc.get("provenance"):
        bad.append("acceptance receipt lists no evaluation provenance")
    return bad


def probe_problems(probe_p, receipt_p, mdir, plan, tier_key):
    """Probe receipt validation (Codex I2-b): phases cover prefill and every selected method of the block, finite
    non-negative measurements, positive capacity, top-level peak = max over phases, peak x 1.1 <= capacity; the
    sidecar receipt written right after the probe binds it to the current weights (full sha256, size, mtime), code and parameters."""
    bad = []
    pc, tier = plan["primary_cell"], plan["tiers"][tier_key]
    try:
        r = json.load(open(probe_p))
    except (OSError, ValueError):
        return None, [f"probe {probe_p} unreadable"]
    blk = r.get("block")
    meths = sorted({plan["arms"][n]["method"] for n in tier["arms"] if plan["arms"][n]["block"] == blk})
    if not meths:
        bad.append(f"probe {probe_p}: block {blk} not used by the tier")
    if os.path.realpath(str(r.get("model", ""))) != os.path.realpath(mdir) or r.get("length") != pc["length"] \
            or r.get("budget") != pc["budget"]:
        bad.append(f"probe {probe_p} is for another model / length / budget")
    ph = r.get("phases")
    need = ["prefill"] + [f"decode:{m}" for m in meths]
    if not isinstance(ph, dict) or any(k not in ph for k in need):
        bad.append(f"probe {probe_p}: phases missing {[k for k in need if not isinstance(ph, dict) or k not in ph]}")
        return blk, bad
    peaks = []
    for k, v in ph.items():
        vals = [v.get(x) if isinstance(v, dict) else None for x in ("peak_alloc_gib", "peak_reserved_gib")]
        if any(not isinstance(x, (int, float)) or x != x or x < 0 or x == float("inf") for x in vals):
            bad.append(f"probe {probe_p}: phase {k} has non-finite or negative measurements")
        else:
            peaks.append(vals[1])
    cap = r.get("gpu_total_gib")
    if not isinstance(cap, (int, float)) or not cap > 0 or cap == float("inf"):
        bad.append(f"probe {probe_p}: device capacity missing or not positive")
    if peaks and isinstance(cap, (int, float)) and cap > 0:
        peak = max(peaks)
        if "peak_reserved_gib" in r and abs(r["peak_reserved_gib"] - peak) > 1e-9:
            bad.append(f"probe {probe_p}: top-level peak {r['peak_reserved_gib']} != max over phases {peak}")
        if peak * 1.1 > cap:
            bad.append(f"probe {probe_p}: peak {peak:.2f} GiB x 1.1 exceeds the device capacity {cap:.2f} GiB")
    try:
        rc = json.load(open(receipt_p))
    except (OSError, ValueError):
        return blk, bad + [f"probe receipt {receipt_p} missing or unreadable"]
    try:
        files = weight_files(mdir)
    except ValueError as e:
        return blk, bad + [str(e)]
    now = {f: file_record(mdir, f) for f in files}   # full sha256 kept (Codex I2-b)
    if rc.get("probe_sha256") != sha(probe_p) or rc.get("weights") != now \
            or rc.get("code") != {c: sha(os.path.join(ROOT, c)) for c in PROBE_CODE} \
            or rc.get("params") != dict(length=pc["length"], block=blk, budget=pc["budget"], methods=meths, layout_unit=32):
        bad.append(f"probe receipt {receipt_p} does not bind this probe to the current weights / code / parameters")
    return blk, bad


def probe_receipt(a):
    """Written by the prep script right after each probe run (binds provenance; Codex I2-b)."""
    plan = json.load(open(PLAN2_PATH))
    r = json.load(open(a.probe))
    mdir = os.path.join(a.models_dir, a.model)
    blk = r.get("block")
    tier = plan["tiers"][a.tier]
    meths = sorted({plan["arms"][n]["method"] for n in tier["arms"] if plan["arms"][n]["block"] == blk})
    try:
        files = weight_files(mdir)
    except ValueError as e:
        print(f"PROBE RECEIPT REFUSED: {e}")
        sys.exit(1)
    json.dump(dict(probe_sha256=sha(a.probe), weights={f: file_record(mdir, f)
                                                       for f in files},
                   code={c: sha(os.path.join(ROOT, c)) for c in PROBE_CODE},
                   params=dict(length=plan["primary_cell"]["length"], block=blk, budget=plan["primary_cell"]["budget"],
                               methods=meths, layout_unit=32)),
              open(a.probe + ".receipt.json", "w"), indent=1, sort_keys=True)


def register(a):
    plan = json.load(open(PLAN2_PATH))
    tier, pc = plan["tiers"][a.tier], plan["primary_cell"]
    mdir = os.path.join(a.models_dir, a.model)
    bad = []
    if a.model not in tier["models"]:
        bad.append(f"model {a.model} not in tier {a.tier}")
    if not HEX40.match(a.revision or ""):
        bad.append("revision must be a 40-hex HF commit")
    if a.hf_repo != plan["models"].get(a.model, {}).get("hf_id"):
        bad.append(f"hf repo {a.hf_repo} != plan {plan['models'].get(a.model, {}).get('hf_id')}")
    try:
        files = weight_files(mdir)
    except ValueError as e:
        bad.append(str(e))
        files = []
    data_dir = os.path.join(a.out_root, "data", a.model, str(pc["length"]))
    decision = None
    if files:
        decision, abad, rep, _ = acceptance_problems(a.adapter_report, a.adapter_verdict, a.adapter_tie_diag,
                                                     a.adapter_pred_root, mdir, plan)
        bad += abad
        rbad = acceptance_receipt_problems(a, mdir, plan, None)
        bad += rbad
        if rep is not None:                  # re-derive linkage and evidence now as well (also names the reason)
            _, ibad = acceptance_inputs(a.adapter_pred_root, a.adapter_data_dir, plan, plan["acceptance"]["order_task"],
                                        rep, mdir)
            bad += ibad
            bad += evidence_problems(a.adapter_pred_root, a.adapter_data_dir, plan["acceptance"]["order_task"], rep,
                                     a.adapter_report, a.adapter_verdict, a.adapter_tie_diag, mdir)
    else:
        bad.append("model files incomplete")
    blocks = sorted({plan["arms"][n]["block"] for n in tier["arms"]})
    seen = set()
    for p in a.probe:
        blk, pbad = probe_problems(p, p + ".receipt.json", mdir, plan, a.tier)
        bad += pbad
        seen.add(blk)
    if seen != set(blocks):
        bad.append(f"probes cover blocks {sorted(x for x in seen if x is not None)}, tier needs {blocks}")
    data_dir = os.path.join(a.out_root, "data", a.model, str(pc["length"]))
    prep_p = os.path.join(data_dir, "prep_manifest.json")
    if not os.path.exists(prep_p) or not os.path.exists(a.units):
        bad.append("prep manifest or unit file missing")
    if bad:
        print("REGISTER REFUSED:", *bad, sep="\n  ")
        sys.exit(1)
    prep = json.load(open(prep_p))
    sys.path.insert(0, HERE)
    import make_jobs_v2 as MJ
    rd = dict(schema="twin-grid v2 run descriptor 1", created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), tier=a.tier,
              plan_v2_sha256=sha(PLAN2_PATH), plan_status=plan.get("status"),
              model=dict(key=a.model, hf_repo=a.hf_repo, revision=a.revision,
                         files={f: file_record(mdir, f) for f in files}),
              data=dict(prep_manifest_sha256=sha(prep_p), tasks={t: o["sha256"] for t, o in prep["outputs"].items()}),
              units_sha256=sha(a.units), code=code_hashes(),
              adapter=dict(verdict=decision, verdict_sha256=sha(a.adapter_verdict), report_sha256=sha(a.adapter_report),
                           tie_diagnostics={os.path.basename(p): sha(p) for p in a.adapter_tie_diag},
                           code={c: sha(os.path.join(ROOT, c)) for c in ACCEPT_CODE},
                           receipt_sha256=sha(a.adapter_report + ".receipt.json")),
              probes={str(json.load(open(p))["block"]): dict(sha256=sha(p), receipt_sha256=sha(p + ".receipt.json"))
                      for p in a.probe},
              matrix=[list(map(str, g[:6])) + [",".join(g[6])] for g in MJ.expand(plan, a.tier) if g[0] == a.model])
    json.dump(rd, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"registered {a.out} sha256 {sha(a.out)}")


def check(rd_path):
    """Schema + current plan/code identity; returns (rd, problems)."""
    bad = []
    try:
        rd = json.load(open(rd_path))
    except (OSError, ValueError):
        return None, ["run descriptor missing or unreadable"]
    if rd.get("schema") != "twin-grid v2 run descriptor 1":
        bad.append("run descriptor schema")
    m = rd.get("model") or {}
    files = m.get("files") or {}
    if not HEX40.match(str(m.get("revision", ""))):
        bad.append("descriptor revision is not 40 hex")
    if any(not HEX64.match(str(v.get("sha256", ""))) for v in files.values()):
        bad.append("descriptor weight hash not a sha256")
    if not any(f.endswith(".safetensors") for f in files) or any(f not in files for f in META):
        bad.append("descriptor weight file set incomplete")
    if rd.get("plan_v2_sha256") != sha(PLAN2_PATH):
        bad.append("plan_v2.json changed since the run was registered")
    cur = code_hashes()
    for c in CODE:                                  # every listed file, the gate and this file included
        if (rd.get("code") or {}).get(c) != cur[c]:
            bad.append(f"code {c} differs from the registered hash (or is not recorded)")
    if not str((rd.get("adapter") or {}).get("verdict", "")).startswith("ACCEPTED"):
        bad.append("descriptor adapter verdict not ACCEPTED")
    return rd, bad


def manifest_vs_rd(man, rd):
    """The manifest's model/code ids must be the registered ones (prefix of the full sha256; shards by size/mtime)."""
    bad, code, files = [], man.get("code") or {}, rd["model"]["files"]
    for f in ("src/ruler_pred.py", "src/sparse_attn.py"):
        if code.get(os.path.basename(f)) != rd["code"][f][:16]:
            bad.append(f"manifest {os.path.basename(f)} id differs from the registered code")
    for f, rec in files.items():
        v = code.get(f"model:{f}")
        if f.endswith(".safetensors"):
            if v != f"size={rec['size']} mtime={rec['mtime']}":
                bad.append(f"manifest model:{f} differs from the registered weights")
        elif v != rec["sha256"][:16]:
            bad.append(f"manifest model:{f} differs from the registered file")
    if code.get("model:revision") != rd["model"]["revision"]:
        bad.append("manifest model revision differs from the registered revision")
    extra = [k for k in code if k.startswith("model:") and k[6:] not in files and k != "model:revision"]
    if extra:
        bad.append(f"manifest lists weight files not in the descriptor: {extra}")
    return bad


def verify(a):
    rd, bad = check(a.rd)
    if rd is None:
        print("VERIFY FAILED:", *bad, sep="\n  ")
        sys.exit(1)
    mdir = os.path.join(a.models_dir, rd["model"]["key"])
    try:
        files = weight_files(mdir)
        if sorted(files) != sorted(rd["model"]["files"]):
            bad.append("weight file set differs from the descriptor")
        for f in files:
            if f in rd["model"]["files"] and file_record(mdir, f) != rd["model"]["files"][f]:
                bad.append(f"weight file {f} differs from the descriptor")
    except ValueError as e:
        bad.append(str(e))
    for c, h in ((rd.get("adapter") or {}).get("code") or {}).items():
        if sha(os.path.join(ROOT, c)) != h:
            bad.append(f"acceptance code {c} changed since registration")
    plan = json.load(open(PLAN2_PATH))
    data_dir = os.path.join(a.out_root, "data", rd["model"]["key"], str(plan["primary_cell"]["length"]))
    prep_p = os.path.join(data_dir, "prep_manifest.json")
    if not os.path.exists(prep_p) or sha(prep_p) != rd["data"]["prep_manifest_sha256"]:
        bad.append("prep manifest differs from the descriptor")
    for t, h in rd["data"]["tasks"].items():
        f = os.path.join(data_dir, t, "validation.jsonl")
        if not os.path.exists(f) or sha(f) != h:
            bad.append(f"{t}: data differs from the descriptor")
    units = os.path.join(data_dir, f"unit_lengths_b32_n{plan['unit_file']['n']}.json")
    if not os.path.exists(units) or sha(units) != rd["units_sha256"]:
        bad.append("unit file differs from the descriptor")
    if bad:
        print("VERIFY FAILED:", *bad, sep="\n  ")
        sys.exit(1)
    print(f"VERIFIED {a.rd} sha256 {sha(a.rd)}")


def bind(a):
    rd, bad = check(a.rd)
    if rd is None or bad:
        print("BIND REFUSED:", *bad, sep="\n  ")
        sys.exit(1)
    man_p = os.path.join(a.save_dir, f"manifest.{a.task}.json")
    man = json.load(open(man_p))
    bad = manifest_vs_rd(man, rd)
    preds = {}
    for m in a.methods.split(","):
        p = os.path.join(a.save_dir, f"{m}@{a.budget}", f"{a.task}.jsonl")
        if not os.path.exists(p):
            bad.append(f"prediction file missing: {p}")
        else:
            preds[m] = sha(p)
    if bad:
        print("BIND REFUSED:", *bad, sep="\n  ")
        sys.exit(1)
    json.dump(dict(rd_sha256=sha(a.rd), task=a.task, manifest_sha256=sha(man_p), predictions=preds),
              open(os.path.join(a.save_dir, f"binding.{a.task}.json"), "w"), indent=1, sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    for k in ("tier", "model", "models-dir", "hf-repo", "revision", "out-root", "units", "adapter-verdict",
              "adapter-report", "adapter-pred-root", "adapter-data-dir", "out"):
        r.add_argument(f"--{k}", required=True)
    r.add_argument("--adapter-tie-diag", action="append", default=[], help="one accept_tie_diag.py JSON per mismatch")
    r.add_argument("--probe", action="append", default=[], required=True,
                   help="probe JSON; its receipt <probe>.receipt.json must exist (probe-receipt)")
    pr = sub.add_parser("probe-receipt")
    for k in ("probe", "tier", "model", "models-dir"):
        pr.add_argument(f"--{k}", required=True)
    ar = sub.add_parser("accept-receipt")
    for k in ("pred-root", "data-dir", "order-task", "report", "verdict", "model", "models-dir"):
        ar.add_argument(f"--{k}", required=True)
    ar.add_argument("--tie-diag", action="append", default=[])
    ev = sub.add_parser("eval-run")
    for k in ("out", "model", "models-dir"):
        ev.add_argument(f"--{k}", required=True)
    ev.add_argument("--watch", action="append", default=[], required=True, help="input directory or file (repeatable)")
    ev.add_argument("--stdout-to-out", action="store_true", help="the evaluator's stdout becomes the output file")
    ev.add_argument("evaluator", nargs=argparse.REMAINDER, help="-- evaluator command")
    v = sub.add_parser("verify")
    for k in ("rd", "models-dir", "out-root"):
        v.add_argument(f"--{k}", required=True)
    b = sub.add_parser("bind")
    for k in ("rd", "save-dir", "task", "methods", "budget"):
        b.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    {"register": register, "probe-receipt": probe_receipt, "accept-receipt": accept_receipt, "eval-run": eval_run,
     "verify": verify,
     "bind": bind}[a.cmd](a)


if __name__ == "__main__":
    main()
