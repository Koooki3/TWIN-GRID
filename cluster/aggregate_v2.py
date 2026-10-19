"""Plan v2 aggregation for the primary (model, length, budget) cell (docs/PLAN_V2_COGNITIVE_ZH.md, frozen 2026-10-19).
Nothing is computed before the gate passes (exit 2 on any failure). Reuses the reviewed v1.4 checks of
cluster/aggregate.py (imported, unchanged) and adds the v2 requirements.

Gate:
  - RULER: checkout HEAD and every pinned source sha256 as v1.4; prep manifest ruler_commit;
  - data: per task the frozen file and its uid list exactly as v1.4, with >= 200 lines;
  - per (arm, task): directory b<block>/t<tier>_n<limit>, limit = 200 for calibration arms on S_unit tasks, else 100;
    the v1.4 manifest checks (aggregate.manifest_problems: code keys, model revision recorded, data hash, ruler commit,
    tokens_to_generate, special tokens, stop ids, budget/block/limit/method, protocol values, --strict-budget);
    one shared code/model/protocol identity across all arms and tasks; rows: no duplicate or foreign uid, lines
    0..limit-1 present, answers and input_sha equal to the frozen line, sparse budget counters clean; token_sha equal
    across every arm for every uid both share;
  - run descriptor (Codex implementation review I1–I3; cluster/v2_run_descriptor.py): schema, 40-hex revision, full
    sha256 for every weight file, current plan and code hashes equal to the registered ones, ACCEPTED adapter verdict;
    tier and model match; the prep manifest, every task's data and the unit file have the descriptor's sha256 (the
    expectations never come from the audited files' own fields); every manifest's model/code ids equal the
    descriptor's (code prefixes, config/tokenizer/index prefixes, shards by size/mtime, revision; no extra weight
    files); every (dir, task) has a binding written after its job, for this descriptor, with the current manifest and
    prediction sha256 (predictions changed after binding are refused);
  - main / calibration separation: per S_unit task the input sets of lines 0..99 and 100..199 are disjoint;
  - unit file (v2 §3.2): version unit_lengths v3, block 32, unit 32, local 64, tokenizer bound to the manifests,
    per task bound to the frozen data sha256 with >= 200 samples; S_unit samples carry one class per answer; no-unit
    tasks carry no lengths and no classes.
Analysis (registered parameters of plan_v2.json unless --test-params):
  Holm-3 (H1, H2, H3) with sample-level sign-flip permutations; H3 estimable iff main S_unit in0 and repair >= 50;
  descriptive bootstrap intervals with the 1% invalid-draw rule; H4 (cobs; tier >= 2 also mean as auxiliary);
  descriptive tables: per-arm unit accuracy on S_unit, G per arm, unrounded official per-task scores for all 13 tasks.
Output flags: registered_parameters (no --test-params) and formal_v2_result (registered parameters AND plan_v2.json
status FROZEN AND the descriptor registered under that FROZEN plan). A DRAFT plan never yields a formal result.
The JSON receipt binds the descriptor sha256, the sha256 of every prediction file used and of the gate/statistics code.
Usage: aggregate_v2.py --ruler ROOT --out-root DIR --tier T --units FILE --run-descriptor RD.json [--json OUT]
       [--test-params PERM,BOOT]"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
import aggregate as A1  # noqa: E402  (frozen v1.4 checks, imported unchanged)
import v2_run_descriptor as RD  # noqa: E402
import v2_stats as V  # noqa: E402

PLAN2_PATH = os.path.join(HERE, "plan_v2.json")
PLAN2 = json.load(open(PLAN2_PATH))


def gate(a):
    bad = []
    head = subprocess.run(["git", "-C", a.ruler, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != A1.PLAN["ruler_commit"] or head != PLAN2["ruler_commit"]:
        bad.append(f"RULER checkout {head or '?'} != plan {PLAN2['ruler_commit']}")
    for rel, h in A1.PLAN["pinned_ruler_sources"].items():
        p = os.path.join(a.ruler, rel)
        if not os.path.exists(p) or A1.sha(p) != h:
            bad.append(f"RULER source {rel} missing or differs from the pinned sha256")
    if bad:
        return bad, None
    metrics = A1.load_mod(os.path.join(a.ruler, "scripts", "eval", "synthetic", "constants.py"), "ruler_eval_constants")
    tconst = A1.load_mod(os.path.join(a.ruler, "scripts", "data", "synthetic", "constants.py"), "ruler_task_constants")
    import yaml
    cfg = yaml.safe_load(open(os.path.join(a.ruler, "scripts", "synthetic.yaml")))
    pc, tier = PLAN2["primary_cell"], PLAN2["tiers"][a.tier]
    model, L, bud = pc["model"], pc["length"], pc["budget"]
    rd, rbad = RD.check(a.run_descriptor)          # Codex I1-I3: expectations come from the registered descriptor
    bad += [f"run descriptor: {x}" for x in rbad]
    if rd is None:
        return bad, None
    rd_sha = A1.sha(a.run_descriptor)
    if rd.get("tier") != a.tier or rd["model"]["key"] != model:
        bad.append("run descriptor is for another tier or model")
    s_unit = [t for t in PLAN2["tasks"] if t not in PLAN2["no_unit_tasks"]]
    calib_arms = set(tier["calibration_arms"])
    arms = {n: PLAN2["arms"][n] for n in tier["arms"]}
    data_dir = os.path.join(a.out_root, "data", model, str(L))
    prep = json.load(open(os.path.join(data_dir, "prep_manifest.json")))
    if prep.get("ruler_commit") != PLAN2["ruler_commit"]:
        bad.append("prep manifest ruler_commit differs from the plan")
    if A1.sha(os.path.join(data_dir, "prep_manifest.json")) != rd["data"]["prep_manifest_sha256"]:
        bad.append("prep manifest differs from the run descriptor")
    shared, digests, data_fp, frozen, rows_by, fns, pred_sha = None, [], {}, {}, {}, {}, {}
    for task in PLAN2["tasks"]:
        f = os.path.join(data_dir, task, "validation.jsonl")
        o = prep.get("outputs", {}).get(task)
        if not o or not os.path.exists(f) or A1.sha(f) != o["sha256"]:
            bad.append(f"{task}: frozen data missing or changed")
            continue
        if rd["data"]["tasks"].get(task) != o["sha256"]:
            bad.append(f"{task}: data differs from the run descriptor")
            continue
        lines = [json.loads(x) for x in open(f)]
        derived = [f"{o['sha256'][:16]}:{task}:{i}" for i in range(len(lines))]
        if o.get("sample_uids") != derived:
            bad.append(f"{task}: prep uid list is not the re-derived full list")
            continue
        if len(lines) < 200:
            bad.append(f"{task}: frozen file has {len(lines)} < 200 lines")
            continue
        data_fp[task], frozen[task] = o["sha256"], lines
        kind = cfg[task]["task"]
        fns[task] = metrics.TASKS[kind]["metric_fn"]
        tokgen = tconst.TASKS[kind]["tokens_to_generate"]
        tok_ref = {}
        for name, arm in arms.items():
            lim = 200 if (name in calib_arms and task in s_unit) else 100
            d = os.path.join(a.out_root, "pred", model, str(L), f"budget{bud}", f"b{arm['block']}", f"t{a.tier}_n{lim}")
            man_p = os.path.join(d, f"manifest.{task}.json")
            p = os.path.join(d, f"{arm['method']}@{bud}", f"{task}.jsonl")
            if not os.path.exists(man_p) or not os.path.exists(p):
                bad.append(f"{task} {name}: missing manifest or predictions in {d}")
                continue
            man = json.load(open(man_p))
            bad += [f"{task} {name}: {x}" for x in A1.manifest_problems(man, task, arm, lim, bud, o["sha256"], tokgen)]
            bad += [f"{task} {name}: {x}" for x in RD.manifest_vs_rd(man, rd)]
            bp = os.path.join(d, f"binding.{task}.json")
            try:
                bnd = json.load(open(bp))
            except (OSError, ValueError):
                bnd = {}
            psha = A1.sha(p)
            if bnd.get("rd_sha256") != rd_sha or bnd.get("manifest_sha256") != A1.sha(man_p) \
                    or (bnd.get("predictions") or {}).get(arm["method"]) != psha:
                bad.append(f"{task} {name}: binding missing, for another descriptor, or manifest/predictions changed since")
            pred_sha[f"{name}/{task}"] = psha
            ident = json.dumps(dict(code={k: v for k, v in (man.get("code") or {}).items() if k != "data"},
                                    stop_ids=man.get("stop_ids"), special_tokens=man.get("special_tokens")), sort_keys=True)
            if shared is None:
                shared = ident
            elif ident != shared:
                bad.append(f"{task} {name}: code/model/protocol identity differs from the other arms or tasks")
            digests.append(A1.sha(man_p))
            rows = [json.loads(x) for x in open(p)]
            uids = [r.get("sample_uid") for r in rows]
            want = derived[:lim]
            if len(set(uids)) != len(uids) or not set(uids) <= set(derived) or not set(want) <= set(uids):
                bad.append(f"{task} {name}: duplicate, foreign or missing sample uids")
                continue
            by = {r["sample_uid"]: r for r in rows}
            ok = True
            for i, u in enumerate(want):
                r, ln = by[u], lines[i]
                if r.get("outputs") != ln["outputs"] or r.get("input_sha") != hashlib.sha256(
                        (ln["input"] + ln.get("answer_prefix", "")).encode()).hexdigest()[:16]:
                    bad.append(f"{task} {name} {u}: answers/input differ from the frozen line")
                    ok = False
                    break
                if arm["method"] != "dense" and not A1.counters_ok(r.get("tokens")):
                    bad.append(f"{task} {name} {u}: budget counters missing, non-integer or non-zero violations")
                    ok = False
                    break
                ts = r.get("token_sha")
                if ts is None or tok_ref.setdefault(u, ts) != ts:
                    bad.append(f"{task} {name} {u}: token_sha missing or differs between arms")
                    ok = False
                    break
            if ok:
                rows_by[(name, task)] = {i: by[u] for i, u in enumerate(want)}
        if task in s_unit:
            sh = lambda ln: hashlib.sha256((ln["input"] + ln.get("answer_prefix", "")).encode()).hexdigest()  # noqa: E731
            if {sh(x) for x in lines[:100]} & {sh(x) for x in lines[100:200]}:
                bad.append(f"{task}: main and calibration inputs overlap")
    # strong model identity: every manifest was compared with the descriptor above (RD.manifest_vs_rd)
    code = json.loads(shared)["code"] if shared else {}
    # unit file: must be the one frozen in the descriptor (not re-derived from its own fields)
    if A1.sha(a.units) != rd["units_sha256"]:
        bad.append("unit file differs from the one frozen in the run descriptor")
    u = json.load(open(a.units))
    uf = PLAN2["unit_file"]
    if u.get("version") != uf["version"] or u.get("block") != uf["block"] or str(u.get("unit")) != str(uf["unit"]) \
            or str(u.get("local")) != str(uf["local"]):
        bad.append("unit file version/block/unit/local differ from plan v2")
    if not code.get("model:tokenizer.json") or not str(u.get("tokenizer_json_sha256", "")).startswith(code["model:tokenizer.json"]):
        bad.append("unit file tokenizer differs from the manifests")
    for t in PLAN2["tasks"]:
        r = (u.get("tasks") or {}).get(t)
        if not r or r.get("data_sha256") != data_fp.get(t) or len(r.get("samples", [])) < 200:
            bad.append(f"{t}: unit file missing, bound to other data, or fewer than 200 samples")
            continue
        for i in range(200):
            smp, ln = r["samples"][i], frozen[t][i]
            cls = smp.get("cls", [])
            if t in PLAN2["no_unit_tasks"]:
                if smp.get("lens") or smp.get("cls"):
                    bad.append(f"{t}: no-unit task has unit lengths or classes")
                    break
            elif len(cls) != len(ln["outputs"]) or any(c not in ("in0", "repair", "out", "fixed") for c in cls):
                bad.append(f"{t} line {i}: unit classes do not match the answers")
                break
    ctx = dict(s_unit=s_unit, arms=arms, calib_arms=calib_arms, rows=rows_by, frozen=frozen, units=u, fns=fns,
               digests=digests, data_fp=data_fp, identity=code, rd=rd, rd_sha=rd_sha, pred_sha=pred_sha)
    return bad, ctx


def records(ctx, arms, lines, tasks):
    recs = []
    for t in tasks:
        for i in lines:
            refs = ctx["frozen"][t][i]["outputs"]
            cls = ctx["units"]["tasks"][t]["samples"][i]["cls"]
            for k, ref in enumerate(refs):
                recs.append(dict(task=t, i=i, cls=cls[k], ok={
                    a: int(ref.lower() in A1.postprocess_pred(ctx["rows"][(a, t)][i]["pred"]).lower()) for a in arms}))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--tier", required=True, choices=["1", "2", "3"])
    ap.add_argument("--units", required=True)
    ap.add_argument("--run-descriptor", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--test-params", default="", help="PERM,BOOT for synthetic tests only (output marked NOT registered)")
    a = ap.parse_args()
    bad, ctx = gate(a)
    if bad:
        print("GATE FAILED:", *bad[:30], sep="\n  ")
        sys.exit(2)
    hf = PLAN2["holm_family"]
    n_perm, n_boot = hf["permutations"], hf["bootstrap"]
    registered = not a.test_params
    if a.test_params:
        n_perm, n_boot = (int(x) for x in a.test_params.split(","))
        print(f"WARNING: --test-params {n_perm},{n_boot}: NOT the registered parameters")
    formal = registered and PLAN2.get("status") == "FROZEN" and ctx["rd"].get("plan_status") == "FROZEN"
    if not formal:
        print("WARNING: NOT a formal v2 result (plan not FROZEN, descriptor not registered under a FROZEN plan, "
              "or non-registered parameters)")
    s_unit, arms = ctx["s_unit"], list(ctx["arms"])
    main_recs = records(ctx, arms, range(100), s_unit)
    est3, n0, nr = V.h3_estimable(main_recs)
    ps, cis, points = {}, {}, {}
    for name, m in hf["members"].items():
        if name == "H3" and not est3:
            ps[name], points[name], cis[name] = None, None, dict(state=f"不可估计（in0 {n0}, repair {nr}）")
            continue
        fn = V.diff if m["stat"] == "diff" else V.gdiff
        d = V.contributions(main_recs, m["a"], m["b"], m["stat"])
        ps[name], points[name] = V.perm_p(d, m["seed_offset"], n_perm=n_perm)
        cis[name] = V.status(V.boot_stat(main_recs, lambda u, fn=fn, m=m: fn(u, m["a"], m["b"]), n_boot=n_boot))
    hol = V.holm(ps, alpha=hf["alpha"])
    calib_arms = sorted(ctx["calib_arms"])
    calib_recs = records(ctx, calib_arms, range(100, 200), s_unit)
    main_c = [dict(r, ok={k: r["ok"][k] for k in calib_arms}) for r in main_recs]
    h4 = {"cobs": V.h4(main_c, calib_recs, s_unit, "cobs", "cobs+g2f", n_boot=n_boot)}
    if {"mean", "mean+g2f"} <= set(calib_arms):
        h4["mean (auxiliary)"] = V.h4(main_c, calib_recs, s_unit, "mean", "mean+g2f", n_boot=n_boot)
    desc = dict(acc={x: V.acc(main_recs, x) for x in arms}, G={x: V.G(main_recs, x) for x in arms},
                official={t: {x: float(np.mean([A1.unrounded(ctx["fns"][t].__name__, A1.postprocess_pred(
                    ctx["rows"][(x, t)][i]["pred"]), ctx["frozen"][t][i]["outputs"]) for i in range(100)])) for x in arms}
                    for t in PLAN2["tasks"]})
    print(f"GATE PASSED. registered parameters: {registered}; formal v2 result: {formal}. "
          f"S_unit main: {len(main_recs)} answer units; calibration: {len(calib_recs)}")
    for name in hf["members"]:
        p, ci = points.get(name), cis[name]
        print(f"{name}: point {p if p is None else round(p, 4)}  p {ps[name]}  Holm {hol[name]['p_holm']:.4g} -> "
              f"{hol[name]['state']}  CI {ci.get('state')} [{ci.get('lo')}, {ci.get('hi')}]")
    for k, r in h4.items():
        print(f"H4 {k} (exploratory): C={r['C']} state={r.get('state')} abs: {r.get('abs_label')} | rel: {r.get('rel_label')}")
    if a.json:
        here = os.path.dirname(os.path.abspath(__file__))
        gate_code = {f: A1.sha(os.path.join(here, f)) for f in ("aggregate_v2.py", "aggregate.py", "v2_run_descriptor.py",
                                                                "plan_v2.json")}
        gate_code["v2_stats.py"] = A1.sha(V.__file__)
        json.dump(dict(registered_parameters=registered, formal_v2_result=formal, run_descriptor_sha256=ctx["rd_sha"],
                       plan_v2_sha256=A1.sha(PLAN2_PATH), units_sha256=A1.sha(a.units), tier=a.tier,
                       gate_and_statistics_code=gate_code, prediction_files=ctx["pred_sha"],
                       manifests=sorted(ctx["digests"]), data=ctx["data_fp"], code_identity=ctx["identity"],
                       holm=hol, points=points, ci=cis, h4=h4, descriptive=desc),
                  open(a.json, "w"), ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    main()
