"""GPU acceptance of src/ruler_pred.py (plan §3.2; Codex R1/R3, batch-1 A2, revision-review Q2). v3.
v3 additions (Q2): the uid list is re-derived from the frozen file ("<sha16>:<task>:<i>" per line) and must equal the
prep manifest's; every manifest must carry the plan's required code keys and fields (missing never "agrees"); per run
kind the protocol is checked (dense: default --local 64; order/single: --local 16; all: --budget, --block 32,
--strict-budget, no --no-special-tokens, special_tokens True, tokens_to_generate = official, non-empty stop_ids equal
across runs); token_sha must be identical for a uid across every run of that task; sparse rows need integer budget
counters with zero violations; the reference end is bound to the predictions by recomputing the model identity of
--model and comparing it with the manifests; coverage is recorded per run (ab / ba / single) and re-derived from the
row's n_input_tokens, n_gen and the manifest's --local/--block, and must agree with the recorded gen_in_candidates.
Inputs are validated before any comparison (exit 1 on any problem):
  - expected samples = the prep manifest's full sample_uid list for each task (all lines of the frozen file; the data
    file's sha256 must equal the manifest's); every prediction file must contain exactly those uids, no duplicates,
    no foreign uids, non-empty;
  - every prediction row's input_sha equals sha256(input + answer_prefix)[:16] of the frozen line and its outputs equal
    the frozen answers;
  - every prediction directory has manifest.<task>.json whose data hash equals the frozen file's and whose argv names
    the expected methods (in the expected order) and budget; all runs share one code/model identity.
Step 1 (official reference, same inputs): RULER's pred/model_wrappers.py HuggingFaceModel with call_api.py's hf
arguments for greedy decoding (do_sample False, repetition_penalty 1, temperature 0.0, top_k 32, top_p 1.0, stop [],
max_new_tokens = tokens_to_generate) on prompt = input + answer_prefix; per uid the official postprocessed text and the
official per-sample score must equal this project's dense run. The path taken (pipeline / generate fallback) is
recorded; differences are reported, never tuned away.
Step 2 (order independence, same-family pairs whose summaries share cache keys, e.g. cobs / cobs+g2f and
quest / quest+g2f): for each pair (A, B), each method's text must be identical across the A→B run, the B→A run and its
single-arm run. Coverage: rows with gen_in_candidates (generated tokens inside a selectable candidate block) are
counted; if no row in a pair's runs is covered, that pair is reported "not covered" and the acceptance fails —
a run whose generations never reach candidate blocks cannot test the cache isolation.
Usage: accept_adapter.py --ruler R --model M --data_dir D --tasks T1,T2 --pred_root P --order_task T
       --pairs cobs:cobs+g2f,quest:quest+g2f --budget B --report OUT.json
Layout under P: dense/, single_<method>/, order_ab_<A>__<B>/ (methods A,B), order_ba_<A>__<B>/ (methods B,A)."""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = json.load(open(os.path.join(HERE, "plan_v1.json")))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))


def postprocess_pred(predict_str):            # verbatim from RULER scripts/eval/evaluate.py
    predict_str = predict_str.strip()
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()
    return predict_str


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def counters_ok(t):
    keys = ("calls", "budget_checked_calls", "budget_violations", "max_abs_budget_error", "fixed_region_violations")
    return isinstance(t, dict) and all(k in t and is_int(t[k]) for k in keys) and t["calls"] > 0 \
        and t["budget_checked_calls"] == t["calls"] and t["budget_violations"] == 0 \
        and t["max_abs_budget_error"] == 0 and t["fixed_region_violations"] == 0


def fixed_end(n, local, block):            # same rule as sparse_attn._fixed_end (grid offset 0)
    return block + max(0, (n - local) // block - 1) * block


def load_run(save_dir, method, task, budget, methods_expected, frozen, bad, identities, local="64", tokgen=None,
             stops=None, tokshas=None):
    """Validated {uid: row} for one method in one save dir."""
    man = os.path.join(save_dir, f"manifest.{task}.json")
    p = os.path.join(save_dir, f"{method}@{budget}", f"{task}.jsonl")
    if not os.path.exists(man) or not os.path.exists(p):
        bad.append(f"{save_dir} {method} {task}: missing manifest or predictions")
        return {}
    m = json.load(open(man))
    argv = m.get("argv", [])

    def val(k, default=None):
        return argv[argv.index(k) + 1] if k in argv and argv.index(k) + 1 < len(argv) else default
    if (val("--methods") or "").split(",") != methods_expected or val("--budget") != str(budget) \
            or val("--block") != "32" or val("--local", "64") != local or "--strict-budget" not in argv \
            or "--no-special-tokens" in argv:
        bad.append(f"{save_dir} {task}: manifest argv differs from the protocol ({methods_expected}@{budget}, "
                   f"block 32, local {local}, strict budget, special tokens)")
    code = m.get("code", {})
    miss = [k for k in PLAN["protocol"]["required_code_keys"] if k not in code]
    if miss:
        bad.append(f"{save_dir} {task}: manifest lacks code keys {miss}")
    if code.get("data") != frozen["sha"][:16]:
        bad.append(f"{save_dir} {task}: manifest data hash differs from the frozen file")
    if m.get("special_tokens") is not True or (tokgen is not None and m.get("tokens_to_generate") != tokgen) \
            or not isinstance(m.get("stop_ids"), list) or not m.get("stop_ids"):
        bad.append(f"{save_dir} {task}: special_tokens / tokens_to_generate / stop_ids not per protocol")
    if stops is not None:
        stops.add(json.dumps(m.get("stop_ids")))
    identities.add(json.dumps({k: v for k, v in code.items() if k != "data"}, sort_keys=True))
    rows = [json.loads(l) for l in open(p)]
    uids = [r.get("sample_uid") for r in rows]
    if not rows or len(set(uids)) != len(uids) or set(uids) != set(frozen["uids"]):
        bad.append(f"{save_dir} {method} {task}: uids not exactly the frozen set (n={len(rows)})")
        return {}
    out = {}
    for r in rows:
        ln = frozen["by_uid"][r["sample_uid"]]
        if r.get("input_sha") != hashlib.sha256((ln["input"] + ln.get("answer_prefix", "")).encode()).hexdigest()[:16] \
                or r.get("outputs") != ln["outputs"]:
            bad.append(f"{save_dir} {method} {r['sample_uid']}: input/answers differ from the frozen line")
        if method != "dense" and not counters_ok(r.get("tokens")):
            bad.append(f"{save_dir} {method} {r['sample_uid']}: budget counters missing/non-integer/violations")
        if tokshas is not None:
            tokshas.setdefault(r["sample_uid"], set()).add(r.get("token_sha"))
        if "gen_in_candidates" in r:
            L, g = r.get("n_input_tokens"), r.get("n_gen")
            if not (is_int(L) and is_int(g)) or \
                    (fixed_end(L + g - 1, int(val("--local", "64")), int(val("--block"))) > L) != r["gen_in_candidates"]:
                bad.append(f"{save_dir} {method} {r['sample_uid']}: gen_in_candidates does not re-derive")
        out[r["sample_uid"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    for k in ("ruler", "model", "data_dir", "tasks", "pred_root", "order_task", "pairs", "report"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--budget", type=int, required=True)
    a = ap.parse_args()
    scripts = os.path.join(a.ruler, "scripts")
    prep = json.load(open(os.path.join(a.data_dir, "prep_manifest.json")))
    tasks = a.tasks.split(",")
    frozen, bad, identities = {}, [], set()
    for t in tasks + [a.order_task]:
        f = os.path.join(a.data_dir, t, "validation.jsonl")
        o = prep["outputs"][t]
        if sha(f) != o["sha256"]:
            bad.append(f"{t}: frozen data file changed since preparation")
        lines = [json.loads(l) for l in open(f)]
        uids = [f"{o['sha256'][:16]}:{t}:{i}" for i in range(len(lines))]
        if o.get("sample_uids") != uids:
            bad.append(f"{t}: prep uid list is not the re-derived full list")
        frozen[t] = dict(sha=o["sha256"], uids=uids, by_uid=dict(zip(uids, lines)), lines=lines)
    sys.path.insert(0, os.path.join(scripts, "data", "synthetic"))
    consts0 = load_mod(os.path.join(scripts, "data", "synthetic", "constants.py"), "ruler_task_consts0")
    cfg0 = yaml.safe_load(open(os.path.join(scripts, "synthetic.yaml")))
    tokgen = {t: consts0.TASKS[cfg0[t]["task"]]["tokens_to_generate"] for t in set(tasks + [a.order_task])}
    stops, tokshas = set(), {}
    dense = {t: load_run(os.path.join(a.pred_root, "dense"), "dense", t, a.budget, ["dense"], frozen[t], bad, identities,
                         "64", tokgen[t], stops, tokshas) for t in tasks}
    pairs = [tuple(x.split(":")) for x in a.pairs.split(",")]
    order = {}
    for A, B in pairs:
        tag = f"{A}__{B}"
        for m in (A, B):
            kw = dict(local="16", tokgen=tokgen[a.order_task], stops=stops, tokshas=tokshas)
            order[(tag, m)] = dict(
                ab=load_run(os.path.join(a.pred_root, f"order_ab_{tag}"), m, a.order_task, a.budget, [A, B],
                            frozen[a.order_task], bad, identities, **kw),
                ba=load_run(os.path.join(a.pred_root, f"order_ba_{tag}"), m, a.order_task, a.budget, [B, A],
                            frozen[a.order_task], bad, identities, **kw),
                single=load_run(os.path.join(a.pred_root, f"single_{m}"), m, a.order_task, a.budget, [m],
                                frozen[a.order_task], bad, identities, **kw))
    if len(identities) > 1:
        bad.append(f"{len(identities)} distinct code/model identities across runs")
    if len(stops) > 1:
        bad.append("stop_ids differ across runs")
    multi = [u for u, v in tokshas.items() if len(v) != 1 or None in v]
    if multi:
        bad.append(f"token_sha missing or differs across runs for {len(multi)} samples")
    import ruler_pred
    ref_model = {k: v for k, v in ruler_pred.model_identity(a.model).items() if k != "model:revision"}
    for ident in identities:
        mine = {k: v for k, v in json.loads(ident).items() if k.startswith("model:") and k != "model:revision"}
        if mine != ref_model:
            bad.append("reference --model identity differs from the predictions' model identity")
    report = dict(input_problems=bad, steps={})
    if bad:
        report["passed"] = False
        json.dump(report, open(a.report, "w"), indent=1)
        print("INPUT VALIDATION FAILED:", *bad[:20], sep="\n  ")
        print("NOT ACCEPTED")
        sys.exit(1)
    ok = True
    sys.path.insert(0, os.path.join(scripts, "pred"))
    wrappers = load_mod(os.path.join(scripts, "pred", "model_wrappers.py"), "ruler_model_wrappers")
    metrics_mod = load_mod(os.path.join(scripts, "eval", "synthetic", "constants.py"), "ruler_metrics")
    sys.path.insert(0, os.path.join(scripts, "data", "synthetic"))
    consts = load_mod(os.path.join(scripts, "data", "synthetic", "constants.py"), "ruler_task_consts")
    cfg = yaml.safe_load(open(os.path.join(scripts, "synthetic.yaml")))
    import torch
    for t in tasks:
        kind = cfg[t]["task"]
        metric = metrics_mod.TASKS[kind]["metric_fn"]
        llm = wrappers.HuggingFaceModel(name_or_path=a.model, do_sample=False, repetition_penalty=1, temperature=0.0,
                                        top_k=32, top_p=1.0, stop=[], max_new_tokens=consts.TASKS[kind]["tokens_to_generate"])
        path = "pipeline" if llm.pipeline is not None else "generate-fallback"
        rows = []
        for uid in frozen[t]["uids"]:
            ln = frozen[t]["by_uid"][uid]
            ref = postprocess_pred(llm(prompt=ln["input"] + ln.get("answer_prefix", ""))["text"][0])
            mine = postprocess_pred(dense[t][uid]["pred"])
            s_ref, s_mine = metric([ref], [ln["outputs"]]), metric([mine], [ln["outputs"]])
            same = mine == ref and s_ref == s_mine
            ok &= same
            rows.append(dict(uid=uid, same_text=mine == ref, score_ref=s_ref, score_ours=s_mine,
                             ref=None if same else ref, ours=None if same else mine))
        report["steps"][f"same_input:{t}"] = dict(official_path=path, n=len(rows),
                                                  equal=sum(r["same_text"] for r in rows), rows=rows)
        del llm
        torch.cuda.empty_cache()
    for (tag, m), runs in order.items():
        uids = frozen[a.order_task]["uids"]
        identical = all(runs["ab"][u]["pred"] == runs["ba"][u]["pred"] == runs["single"][u]["pred"] for u in uids)
        covered = {k: sum(1 for u in uids if runs[k][u].get("gen_in_candidates")) for k in ("ab", "ba", "single")}
        status = "identical" if identical else "DIFFERENT"
        if min(covered.values()) == 0:
            status = "not covered" if identical else status
        ok &= identical and min(covered.values()) > 0
        report["steps"][f"order:{tag}:{m}"] = dict(status=status, n=len(uids), covered_rows=covered,
                                                   n_gen={k: [runs[k][u].get("n_gen") for u in uids]
                                                          for k in ("ab", "ba", "single")})
    report["passed"] = bool(ok)
    json.dump(report, open(a.report, "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in report["steps"].items()},
                     indent=1))
    print("ACCEPTED" if ok else "NOT ACCEPTED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
