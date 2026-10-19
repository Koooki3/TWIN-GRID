"""End-to-end checks of the v2 pipeline on a synthetic out-root (official RULER metric functions from the real pinned
checkout; a fake 2-shard model snapshot; no model is loaded). Exit 1 on failure.
Pipeline: data + unit file -> adapter verdict + probes -> cluster/v2_run_descriptor.py register -> predictions whose
manifests carry the descriptor's ids -> bind every (dir, task) -> cluster/aggregate_v2.py.
Positive: GATE PASSED; registered_parameters False under --test-params; formal_v2_result False (plan DRAFT); H1/H3
rejected; H4 over the 9 S_unit tasks with coverage; receipt carries the descriptor and prediction hashes.
Codex implementation review negatives:
  I1  register refuses: a missing index shard, a non-40-hex revision, a NOT ACCEPTED adapter verdict, a probe without
      headroom, probes missing a block; aggregation refuses: descriptor weight hash not a sha256, descriptor missing a
      shard (manifest lists an extra weight file), descriptor of another revision, manifest shard size/mtime differing.
  I2  verify refuses: weight content changed after registration, garbage unit file, missing model directory; the
      generated predict script starts with verify, binds after every prediction, needs --rd; its prefix (up to the
      first prediction) exits non-zero with a garbage unit file.
  I3  aggregation refuses: a legal class change in the unit file, a common wrong code id in every manifest, a
      prediction changed after binding, bindings for another descriptor, classes on a no-unit task."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = __file__.rsplit("/theory/", 1)[0]
sys.path.insert(0, ROOT + "/cluster")
import v2_run_descriptor as RD  # noqa: E402

PLAN = json.load(open(ROOT + "/cluster/plan_v2.json"))
RULER = os.environ.get("RULER_ROOT", "/home/kooki3/tools/sources/ruler")
PY = sys.executable
PC = PLAN["primary_cell"]
TIER, M = "3", PC["model"]
REV = "0123456789abcdef0123456789abcdef01234567"
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("tc", os.path.join(RULER, "scripts", "data", "synthetic", "constants.py"))
TC = importlib.util.module_from_spec(_s)
_s.loader.exec_module(TC)
import yaml  # noqa: E402
CFG = yaml.safe_load(open(os.path.join(RULER, "scripts", "synthetic.yaml")))
S_UNIT = [t for t in PLAN["tasks"] if t not in PLAN["no_unit_tasks"]]
TOK = dict(calls=5, budget_checked_calls=5, budget_violations=0, max_abs_budget_error=0, fixed_region_violations=0)
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


def cls_of(t, i):
    if t not in S_UNIT:
        return None
    if t in ("niah_single_3", "niah_multikey_3") and i % 5 == 4:
        return "out"
    return "repair" if i % 3 == 0 else "in0"


def correct(arm, c, i):
    if arm == "dense" or c is None:
        return True
    rules = {"cobs": c == "in0" and i % 10 != 1, "cobs+g2f": (c == "in0" and i % 10 != 1) or (c == "repair" and i % 2 == 0),
             "cobs16": c == "in0" or (c == "repair" and i % 4 != 0), "quest32": c == "in0" or (c == "repair" and i % 5 != 0),
             "quest16+g2f": (c != "in0" or i % 7 != 1), "quest8": i % 23 != 1}
    return rules[arm]


def sh16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def model_dir(root, skip_shard=False):
    d = os.path.join(root, "models", M)
    os.makedirs(d, exist_ok=True)
    for f in RD.META:
        open(os.path.join(d, f), "w").write(json.dumps({"f": f}))
    json.dump({"weight_map": {"w1": SHARDS[0], "w2": SHARDS[1]}}, open(os.path.join(d, "model.safetensors.index.json"), "w"))
    for s in SHARDS[: 1 if skip_shard else 2]:
        open(os.path.join(d, s), "wb").write(s.encode() * 10)
    return d


BLOCK_METHODS = {8: ["quest"], 16: ["cobs", "quest+g2f"], 32: ["cobs", "cobs+g2f", "dense", "quest"]}


def model_ids(mdir):
    code = {"sparse_attn.py": sh16(ROOT + "/src/sparse_attn.py"), "ruler_pred.py": sh16(ROOT + "/src/ruler_pred.py")}
    for f in RD.META + ("model.safetensors.index.json",):
        code[f"model:{f}"] = sh16(os.path.join(mdir, f))
    for s in SHARDS:
        if os.path.exists(os.path.join(mdir, s)):
            st = os.stat(os.path.join(mdir, s))
            code[f"model:{s}"] = f"size={st.st_size} mtime={int(st.st_mtime)}"
    return code


ORDER_TASK = PLAN["acceptance"]["order_task"]
GOOD_TIE = dict(input_equal=True, divergence_step=3, ref_logits=[18.75, 18.5], margin=0.25, ulp=RD.ulp_bf16(18.75), swap=True,
                tie=True)


def acceptance(root, mdir, verdict=None, report_mut=None, ties=(), tie_mut=None, acc_code_mut=None, garbage=False,
               layout_mut=None, receipt=True, pre_existing_report=False, during_eval=None, after_eval=None,
               watch_report=None, watch_diag=None, watch_verdict=None):
    """A structurally complete acceptance in the cluster/accept_adapter.py layout on its own 5-row data set (every
    task): dense/ for every task, order_ab / order_ba / single runs for both pairs on the order task; the report rows
    carry the dense run's uids; `ties` = list of (task, row) made mismatches with full tie diagnostics (tie_mut may
    alter or drop them, returning None to drop). The acceptance receipt is written unless receipt=False."""
    acc = os.path.join(root, "accept")
    data = os.path.join(acc, "data")
    pred = os.path.join(acc, "pred")
    os.makedirs(acc, exist_ok=True)
    outs = {}
    for t in PLAN["tasks"]:
        os.makedirs(os.path.join(data, t), exist_ok=True)
        f = os.path.join(data, t, "validation.jsonl")
        open(f, "w").write("".join(json.dumps(dict(index=i, input=f"acc {t} {i}", answer_prefix="", outputs=["a"])) + "\n"
                                   for i in range(5)))
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        outs[t] = dict(sha256=h, sample_uids=[f"{h[:16]}:{t}:{i}" for i in range(5)])
    json.dump(dict(ruler_commit=PLAN["ruler_commit"], outputs=outs), open(os.path.join(data, "prep_manifest.json"), "w"))
    code = model_ids(mdir)
    if acc_code_mut:
        acc_code_mut(code)

    def run(sub, task, methods):
        d = os.path.join(pred, sub)
        for m in methods:
            os.makedirs(os.path.join(d, f"{m}@{PC['budget']}"), exist_ok=True)
            open(os.path.join(d, f"{m}@{PC['budget']}", f"{task}.jsonl"), "w").write(
                "".join(json.dumps(dict(sample_uid=u, pred="a", n_gen=4, gen_in_candidates=i < 3)) + "\n"
                        for i, u in enumerate(outs[task]["sample_uids"])))
        json.dump(dict(code=dict(code, data=outs[task]["sha256"][:16]),
                       argv=["--methods", ",".join(methods), "--budget", str(PC["budget"])]),
                  open(os.path.join(d, f"manifest.{task}.json"), "w"))
    for t in PLAN["tasks"]:
        run("dense", t, ["dense"])
    for a, b in RD.ORDER_PAIRS:
        for sub, ms in ((f"order_ab_{a}__{b}", [a, b]), (f"order_ba_{a}__{b}", [b, a]), (f"single_{a}", [a]), (f"single_{b}", [b])):
            run(sub, ORDER_TASK, ms)
    if layout_mut:
        layout_mut(pred, outs)
    steps = {}
    for t in PLAN["tasks"]:
        rows = [dict(uid=u, same_text=(t, i) not in ties, score_ref=100.0, score_ours=100.0)
                for i, u in enumerate(outs[t]["sample_uids"])]
        steps[f"same_input:{t}"] = dict(official_path="pipeline", n=5, equal=sum(r["same_text"] for r in rows), rows=rows)
    for a, b in RD.ORDER_PAIRS:
        for m in (a, b):
            steps[f"order:{a}__{b}:{m}"] = dict(status="identical", n=5, covered_rows=dict(ab=3, ba=3, single=3))
    rep = dict(input_problems=[], steps=steps, passed=not ties)
    if report_mut:
        report_mut(rep)
    # every output goes through eval-run with a FAKE evaluator (copies a staged file; no model): CPU-only provenance test
    stage = os.path.join(root, "stage")
    os.makedirs(stage, exist_ok=True)
    rp = os.path.join(acc, "r.json")
    open(os.path.join(stage, "r.json"), "w").write("{not json" if garbage else json.dumps(rep))
    if pre_existing_report:
        open(rp, "w").write(json.dumps(rep))
    fake_copy = [PY, "-c", "import shutil, sys; shutil.copy(sys.argv[1], sys.argv[2])"]
    evaluate(root, rp, watch_report(pred, data) if watch_report else [pred, data], fake_copy + [os.path.join(stage, "r.json"), rp],
             during=during_eval(root) if during_eval else None)
    diags = []
    for k, (t, i) in enumerate(ties):
        d = dict(GOOD_TIE, uid=outs[t]["sample_uids"][i])
        if tie_mut:
            d = tie_mut(d)
        if d is None:
            continue
        p = os.path.join(acc, f"tie_{k}.json")
        json.dump(d, open(os.path.join(stage, f"tie_{k}.json"), "w"))
        evaluate(root, p, watch_diag(pred, data) if watch_diag else [pred, data], fake_copy + [os.path.join(stage, f"tie_{k}.json"), p])
        diags.append(p)
    verdict = verdict or ("ACCEPTED_WITH_TIES: 1 bf16 tie(s)" if ties else "ACCEPTED (strict)")
    vp = os.path.join(acc, "v.txt")
    evaluate(root, vp, watch_verdict(rp, diags) if watch_verdict else [rp] + diags, ["printf", "%s\\n", verdict], stdout=True)
    if after_eval:
        after_eval(root)
    if receipt and not garbage:
        accept_receipt(root, pred, data, rp, vp, diags)
    return vp, rp, pred, diags, data


def evaluate(root, out, watch, cmd, stdout=False, during=None):
    """eval-run around a fake evaluator; `during` = a shell snippet run inside the evaluation (to change an input)."""
    if during:
        cmd = ["bash", "-c", during + ' && "$@"', "x"] + cmd
    full = [PY, ROOT + "/cluster/v2_run_descriptor.py", "eval-run", "--out", out, "--model", M,
            "--models-dir", os.path.join(root, "models")] + sum([["--watch", w] for w in watch], []) + \
        (["--stdout-to-out"] if stdout else []) + ["--"] + cmd
    return subprocess.run(full, capture_output=True, text=True)


def accept_receipt(root, pred, data, rp, vp, diags):
    return subprocess.run([PY, ROOT + "/cluster/v2_run_descriptor.py", "accept-receipt", "--pred-root", pred, "--data-dir", data,
                           "--order-task", ORDER_TASK, "--report", rp, "--verdict", vp, "--model", M, "--models-dir",
                           os.path.join(root, "models")] + sum([["--tie-diag", p] for p in diags], []),
                          capture_output=True, text=True)


def probes_for(root, mdir, peak=10.0, blocks=(8, 16, 32), probe_mut=None, receipt=True, stale=False):
    pr = os.path.join(root, "probe")
    os.makedirs(pr, exist_ok=True)
    out = []
    for b in blocks:
        p = os.path.join(pr, f"b{b}.json")
        ph = {"prefill": dict(peak_alloc_gib=peak - 1, peak_reserved_gib=peak)}
        for m in BLOCK_METHODS[b]:
            ph[f"decode:{m}"] = dict(peak_alloc_gib=peak - 2, peak_reserved_gib=peak - 0.5)
        r = dict(model=mdir, length=PC["length"], block=b, budget=PC["budget"], gpu="synthetic", gpu_total_gib=22.0,
                 weights_gib=3.0, phases=ph, peak_reserved_gib=peak)
        if probe_mut:
            probe_mut(r)
        json.dump(r, open(p, "w"))
        if receipt:
            RD.probe_receipt(argparse.Namespace(probe=p, tier=TIER, model=M, models_dir=os.path.join(root, "models")))
            if stale:
                rc = json.load(open(p + ".receipt.json"))
                rc["weights"][SHARDS[0]]["mtime"] -= 100
                json.dump(rc, open(p + ".receipt.json", "w"))
        out.append(p)
    return out


def swap_shard_content(root):
    """Replace a shard's bytes with different bytes of the same length and restore its exact (ns) mtime."""
    p = os.path.join(root, "models", M, SHARDS[0])
    st = os.stat(p)
    open(p, "wb").write(b"Z" * st.st_size)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))


def register(root, rev=REV, out="rd.json", acc_kw=None, probe_kw=None, after_receipts=None):
    mdir = os.path.join(root, "models", M)
    v, r, pred, diags, adata = acceptance(root, mdir, **(acc_kw or {}))
    probes = probes_for(root, mdir, **(probe_kw or {}))
    if after_receipts:                    # drift between the receipts and registration
        after_receipts(root)
    units = os.path.join(root, "data", M, str(PC["length"]), "unit_lengths_b32_n200.json")
    cmd = [PY, ROOT + "/cluster/v2_run_descriptor.py", "register", "--tier", TIER, "--model", M, "--models-dir",
           os.path.join(root, "models"), "--hf-repo", PLAN["models"][M]["hf_id"], "--revision", rev, "--out-root", root,
           "--units", units, "--adapter-verdict", v, "--adapter-report", r, "--adapter-pred-root", pred,
           "--adapter-data-dir", adata, "--out", os.path.join(root, out)]
    for p in diags:
        cmd += ["--adapter-tie-diag", p]
    for p in probes:
        cmd += ["--probe", p]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.returncode, res.stdout + res.stderr


def build_data(root, unit_mut=None):
    data = os.path.join(root, "data", M, str(PC["length"]))
    outs = {}
    for t in PLAN["tasks"]:
        os.makedirs(os.path.join(data, t), exist_ok=True)
        f = os.path.join(data, t, "validation.jsonl")
        lines = [dict(index=i, input=f"ctx {t} {i}", answer_prefix=" Answer:", outputs=["alpha"], length=10) for i in range(200)]
        open(f, "w").write("".join(json.dumps(x) + "\n" for x in lines))
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        outs[t] = dict(sha256=h, sample_uids=[f"{h[:16]}:{t}:{i}" for i in range(200)], lines=lines)
    json.dump(dict(ruler_commit=PLAN["ruler_commit"], outputs={t: {k: v for k, v in o.items() if k != "lines"}
                                                              for t, o in outs.items()}),
              open(os.path.join(data, "prep_manifest.json"), "w"))
    units = dict(version=PLAN["unit_file"]["version"], block=32, unit=32, local=64,
                 tokenizer_json_sha256=hashlib.sha256(open(os.path.join(root, "models", M, "tokenizer.json"), "rb").read()).hexdigest(),
                 tasks={t: dict(data_sha256=outs[t]["sha256"],
                                samples=[dict(lens=([] if cls_of(t, i) is None else [10]),
                                              cls=([] if cls_of(t, i) is None else [cls_of(t, i)])) for i in range(200)])
                        for t in PLAN["tasks"]})
    if unit_mut:
        unit_mut(units)
    json.dump(units, open(os.path.join(data, "unit_lengths_b32_n200.json"), "w"))
    return outs


def build_preds(root, outs, man_mut=None, rd="rd.json"):
    rdj = json.load(open(os.path.join(root, rd)))
    mdir = os.path.join(root, "models", M)
    code = {"sparse_attn.py": rdj["code"]["src/sparse_attn.py"][:16], "ruler_pred.py": rdj["code"]["src/ruler_pred.py"][:16]}
    for f in RD.META + ("model.safetensors.index.json",):
        code[f"model:{f}"] = sh16(os.path.join(mdir, f))
    for s in SHARDS:
        st = os.stat(os.path.join(mdir, s))
        code[f"model:{s}"] = f"size={st.st_size} mtime={int(st.st_mtime)}"
    code["model:revision"] = rdj["model"]["revision"]
    tier = PLAN["tiers"][TIER]
    groups = {}
    for name in tier["arms"]:
        arm = PLAN["arms"][name]
        for t in PLAN["tasks"]:
            lim = 200 if (name in tier["calibration_arms"] and t in S_UNIT) else 100
            groups.setdefault((arm["block"], lim, t), []).append((name, arm["method"]))
    dirs = []
    for (blk, lim, t), members in groups.items():
        d = os.path.join(root, "pred", M, str(PC["length"]), f"budget{PC['budget']}", f"b{blk}", f"t{TIER}_n{lim}")
        man = dict(code=dict(code, data=outs[t]["sha256"][:16]), ruler_commit=PLAN["ruler_commit"],
                   tokens_to_generate=TC.TASKS[CFG[t]["task"]]["tokens_to_generate"], stop_ids=[1, 2], special_tokens=True,
                   argv=["--methods", ",".join(sorted(m for _, m in members)), "--budget", str(PC["budget"]), "--block",
                         str(blk), "--limit", str(lim), "--strict-budget", "--layout-unit", "32"])
        if man_mut:
            man_mut(man)
        for name, meth in members:
            os.makedirs(os.path.join(d, f"{meth}@{PC['budget']}"), exist_ok=True)
            rows = []
            for i in range(lim):
                ln = outs[t]["lines"][i]
                rows.append(dict(sample_uid=outs[t]["sample_uids"][i], token_sha=f"tok{t}{i}", outputs=["alpha"],
                                 input_sha=hashlib.sha256((ln["input"] + ln["answer_prefix"]).encode()).hexdigest()[:16],
                                 pred="it is alpha" if correct(name, cls_of(t, i), i) else "no idea", tokens=dict(TOK)))
            open(os.path.join(d, f"{meth}@{PC['budget']}", f"{t}.jsonl"), "w").write("".join(json.dumps(r) + "\n" for r in rows))
        json.dump(man, open(os.path.join(d, f"manifest.{t}.json"), "w"))
        dirs.append((d, t, ",".join(sorted(m for _, m in members))))
    return dirs


def bind_all(root, dirs, rd="rd.json"):
    for d, t, meths in dirs:
        try:
            RD.bind(argparse.Namespace(rd=os.path.join(root, rd), save_dir=d, task=t, methods=meths, budget=str(PC["budget"])))
        except SystemExit:
            return False
    return True


def aggregate(root, rd="rd.json"):
    r = subprocess.run([PY, ROOT + "/cluster/aggregate_v2.py", "--ruler", RULER, "--out-root", root, "--tier", TIER,
                        "--units", os.path.join(root, "data", M, str(PC["length"]), "unit_lengths_b32_n200.json"),
                        "--run-descriptor", os.path.join(root, rd), "--json", os.path.join(root, "out.json"),
                        "--test-params", "2000,100"], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


bad = []


def check(cond, what, out=""):
    print(("OK   " if cond else "FAIL ") + what)
    if not cond:
        if out:
            print(out[-1200:])
        bad.append(what)


def fresh(unit_mut=None):
    root = tempfile.mkdtemp(prefix="aggv2_", dir=os.environ.get("TMPDIR"))
    model_dir(root)
    outs = build_data(root, unit_mut)
    return root, outs


def full(unit_mut=None, man_mut=None):
    root, outs = fresh(unit_mut)
    rc, out = register(root)
    assert rc == 0, out
    dirs = build_preds(root, outs, man_mut)
    return root, outs, dirs


# ---- positive
root, outs, dirs = full()
check(bind_all(root, dirs), "bind: every (dir, task) bound")
rc, out = aggregate(root)
check(rc == 0 and "GATE PASSED" in out and "formal v2 result: False" in out, "positive: GATE PASSED, not a formal result", out)
if rc == 0:
    o = json.load(open(os.path.join(root, "out.json")))
    check(o["registered_parameters"] is False and o["formal_v2_result"] is False, "positive: flags registered/formal both False")
    check(o["holm"]["H1"]["reject"] and o["holm"]["H3"]["reject"], "positive: H1 and H3 rejected")
    check(sorted(o["h4"]["cobs"]["C"]) == sorted(S_UNIT) and o["h4"]["cobs"]["coverage"]["cells"] == 9, "positive: H4 over 9 tasks with coverage")
    check(o["run_descriptor_sha256"] == hashlib.sha256(open(os.path.join(root, "rd.json"), "rb").read()).hexdigest()
          and len(o["prediction_files"]) == 7 * 13 and "v2_stats.py" in o["gate_and_statistics_code"],
          "positive: receipt binds descriptor, 91 prediction files and gate/statistics code")
# I2 verify and generated script
v = subprocess.run([PY, ROOT + "/cluster/v2_run_descriptor.py", "verify", "--rd", os.path.join(root, "rd.json"),
                    "--models-dir", os.path.join(root, "models"), "--out-root", root], capture_output=True, text=True)
check(v.returncode == 0 and "VERIFIED" in v.stdout, "verify: positive", v.stdout + v.stderr)
g = subprocess.run([PY, ROOT + "/cluster/make_jobs_v2.py", "--phase", "predict", "--tier", TIER, "--rd", os.path.join(root, "rd.json"),
                    "--models-dir", os.path.join(root, "models"), "--out-root", root], capture_output=True, text=True)
lines = [x for x in g.stdout.splitlines() if x and not x.startswith("#") and x != "set -u"]
preds = [k for k, x in enumerate(lines) if "/src/ruler_pred.py" in x]
check(g.returncode == 0 and "v2_run_descriptor.py verify" in lines[0] and preds and
      all("v2_run_descriptor.py bind" in lines[k + 1] for k in preds), "generated predict: verify first, bind after every prediction")
g2 = subprocess.run([PY, ROOT + "/cluster/make_jobs_v2.py", "--phase", "predict", "--tier", TIER, "--models-dir", "/m",
                     "--out-root", "/o"], capture_output=True, text=True)
check(g2.returncode != 0, "generated predict without --rd refused")
prefix = "\n".join(["set -u"] + lines[: preds[0]])
units_p = os.path.join(root, "data", M, str(PC["length"]), "unit_lengths_b32_n200.json")
keep = open(units_p).read()
open(units_p, "w").write("garbage")
pr = subprocess.run(["bash", "-c", prefix], capture_output=True, text=True)
check(pr.returncode != 0, "generated predict prefix (no prediction) exits non-zero with a garbage unit file", pr.stdout)
v2 = subprocess.run([PY, ROOT + "/cluster/v2_run_descriptor.py", "verify", "--rd", os.path.join(root, "rd.json"),
                     "--models-dir", os.path.join(root, "models"), "--out-root", root], capture_output=True, text=True)
check(v2.returncode != 0, "verify: garbage unit file refused")
open(units_p, "w").write(keep)
shard = os.path.join(root, "models", M, SHARDS[1])
st = os.stat(shard)
open(shard, "wb").write(b"X" * st.st_size)
os.utime(shard, (st.st_atime, st.st_mtime))
v3 = subprocess.run([PY, ROOT + "/cluster/v2_run_descriptor.py", "verify", "--rd", os.path.join(root, "rd.json"),
                     "--models-dir", os.path.join(root, "models"), "--out-root", root], capture_output=True, text=True)
check(v3.returncode != 0, "verify: weight content changed (same size, same mtime) refused")
v4 = subprocess.run([PY, ROOT + "/cluster/v2_run_descriptor.py", "verify", "--rd", os.path.join(root, "rd.json"),
                     "--models-dir", os.path.join(root, "nonexistent"), "--out-root", root], capture_output=True, text=True)
check(v4.returncode != 0, "verify: missing model directory refused")
shutil.rmtree(root)

# ---- register: valid tie case passes (I2-a)
r0 = tempfile.mkdtemp(prefix="aggv2_", dir=os.environ.get("TMPDIR"))
model_dir(r0)
build_data(r0)
rc, out = register(r0, acc_kw=dict(ties=[("vt", 2)]))
check(rc == 0 and json.load(open(os.path.join(r0, "rd.json")))["adapter"]["verdict"] == "ACCEPTED_WITH_TIES",
      "register: valid supported tie case accepted (ACCEPTED_WITH_TIES re-derived)", out)
tp = subprocess.run([PY, ROOT + "/cluster/v2_run_descriptor.py", "tie-plan", "--report", os.path.join(r0, "accept", "r.json"),
                     "--data-dir", os.path.join(r0, "accept", "data"), "--pred-root", os.path.join(r0, "accept", "pred"),
                     "--ruler", RULER], capture_output=True, text=True)
tl = [x.split() for x in tp.stdout.splitlines()]
check(tp.returncode == 0 and len(tl) == 1 and tl[0][1:3] == ["vt", "2"] and tl[0][5] == str(TC.TASKS[CFG["vt"]["task"]]["tokens_to_generate"]),
      "tie-plan: one line for the vt mismatch (line 2, official max_new)", tp.stdout + tp.stderr)
shutil.rmtree(r0)
r0 = tempfile.mkdtemp(prefix="aggv2_", dir=os.environ.get("TMPDIR"))
model_dir(r0)
build_data(r0)
rc, out = register(r0, acc_kw=dict(ties=[("vt", 2)], tie_mut=lambda d: dict(d, divergence_step=0, ref_logits=[0.0, 0.0],
                                                                             margin=0.0, ulp=2.0 ** -133)))
check(rc == 0, "register: zero-logit tie diagnostic (ulp 2^-133, as the frozen producer) accepted", out)
shutil.rmtree(r0)


def pop_task(rep):
    rep["steps"].pop("same_input:qa_1")


def pop_order(rep):
    rep["steps"].pop("order:quest__quest+g2f:quest+g2f")


def failed_report(rep):
    rep["passed"] = False
    rep["input_problems"] = ["reference model differs"]


def phase_only_dense(r):
    r["phases"] = {k: v for k, v in r["phases"].items() if k in ("prefill", "decode:dense")}


def inconsistent_peak(r):
    r["phases"]["prefill"]["peak_reserved_gib"] = 30.0      # top-level stays 10, capacity 22


def only_unrelated(pred, outs):
    shutil.rmtree(pred)
    os.makedirs(pred)
    json.dump(dict(code={}), open(os.path.join(pred, "manifest.unrelated.json"), "w"))


def wrong_data(pred, outs):
    p = os.path.join(pred, "dense", "manifest.vt.json")
    m = json.load(open(p))
    m["code"]["data"] = "0" * 16
    json.dump(m, open(p, "w"))


def drop_order_ba(pred, outs):
    shutil.rmtree(os.path.join(pred, "order_ba_cobs__cobs+g2f"))


def foreign_uid(rep):
    rep["steps"]["same_input:niah_single_2"]["rows"][0]["uid"] = "0" * 16 + ":niah_single_2:0"


def _edit_rows(p, fn):
    rows = [json.loads(x) for x in open(p)]
    for r in rows:
        fn(r)
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))


def order_ba_differs(pred, outs):
    p = os.path.join(pred, "order_ba_cobs__cobs+g2f", f"cobs@{PC['budget']}", f"{ORDER_TASK}.jsonl")
    rows = [json.loads(x) for x in open(p)]
    rows[1]["pred"] = "DIFFERENT ORDER OUTPUT"
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))


def no_coverage(pred, outs):
    for dp, _, fs in os.walk(pred):
        if "order_" in dp or "single_" in dp:
            for f in fs:
                if f.endswith(".jsonl"):
                    _edit_rows(os.path.join(dp, f), lambda r: r.__setitem__("gen_in_candidates", False))


def report_zero_cov(rep):
    for k, v in rep["steps"].items():
        if k.startswith("order:"):
            v["covered_rows"] = dict(ab=0, ba=0, single=0)
            v["status"] = "not covered"


def report_cov_2(rep):
    rep["steps"]["order:cobs__cobs+g2f:cobs"]["covered_rows"] = dict(ab=2, ba=3, single=3)


def CHANGE_DURING(root):          # a shell snippet that appends to a watched acceptance input during the evaluation
    return "echo x >> " + os.path.join(root, "accept", "pred", "dense", f"dense@{PC['budget']}", "vt.jsonl")


def change_dense_vt(root):
    _edit_rows(os.path.join(root, "accept", "pred", "dense", f"dense@{PC['budget']}", "vt.jsonl"),
               lambda r: r.__setitem__("pred", "changed after the evaluation"))


def change_and_rereceipt(root):
    acc = os.path.join(root, "accept")
    p = os.path.join(acc, "pred", "order_ba_cobs__cobs+g2f", f"cobs@{PC['budget']}", f"{ORDER_TASK}.jsonl")
    rows = [json.loads(x) for x in open(p)]
    rows[1]["pred"] = "DIFFERENT ORDER OUTPUT"
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
    r = accept_receipt(root, os.path.join(acc, "pred"), os.path.join(acc, "data"), os.path.join(acc, "r.json"),
                       os.path.join(acc, "v.txt"), [])
    check(r.returncode != 0 and "already exists" in r.stdout, "accept-receipt re-run after drift refuses (no overwrite)", r.stdout)


def rewrite_report(root):
    p = os.path.join(root, "accept", "r.json")
    r = json.load(open(p))
    r["note"] = "rewritten after the acceptance receipt"
    json.dump(r, open(p, "w"))


# ---- register refusals (I1, I2-a, I2-b): (name, model kwargs, register kwargs, required reason)
REFUSALS = (
    ("missing index shard", dict(skip_shard=True), dict(probe_kw=dict(receipt=False)), "that are missing"),
    ("revision not 40 hex", {}, dict(rev="not-a-revision"), "40-hex"),
    ("adapter report not JSON", {}, dict(acc_kw=dict(garbage=True)), "not a JSON object"),
    ("adapter report with input problems / passed false", {}, dict(acc_kw=dict(report_mut=failed_report)), "input problems"),
    ("adapter report with empty steps", {}, dict(acc_kw=dict(report_mut=lambda r: r.__setitem__("steps", {}))), "no steps"),
    ("adapter same-input step missing for a task", {}, dict(acc_kw=dict(report_mut=pop_task)), "same-input step for qa_1"),
    ("adapter order step missing for a pair", {}, dict(acc_kw=dict(report_mut=pop_order)), "order / cache-isolation step"),
    ("stale strict verdict beside a mismatched report", {}, dict(acc_kw=dict(ties=[("vt", 2)], verdict="ACCEPTED (strict)")),
     "does not match the decision"),
    ("mismatch without tie evidence", {}, dict(acc_kw=dict(ties=[("vt", 2)], tie_mut=lambda d: None)), "no tie diagnostic"),
    ("consistent non-tie diagnostic (swap false, tie false)", {},
     dict(acc_kw=dict(ties=[("vt", 2)], tie_mut=lambda d: dict(d, swap=False, tie=False))), "not a tie"),
    ("tie = true but input_equal = false", {}, dict(acc_kw=dict(ties=[("vt", 2)], tie_mut=lambda d: dict(d, input_equal=False))),
     "tie flag contradicts"),
    ("tie = true but swap = false", {}, dict(acc_kw=dict(ties=[("vt", 2)], tie_mut=lambda d: dict(d, swap=False))),
     "tie flag contradicts"),
    ("tie = true with margin 10 / ulp .001 (inconsistent fields)", {},
     dict(acc_kw=dict(ties=[("vt", 2)], tie_mut=lambda d: dict(d, margin=10.0, ulp=0.001))), "margin != l1 - l2"),
    ("tie = true with a consistent margin of 10 logits", {},
     dict(acc_kw=dict(ties=[("vt", 2)], tie_mut=lambda d: dict(d, ref_logits=[20.0, 10.0], margin=10.0, ulp=RD.ulp_bf16(20.0)))),
     "tie flag contradicts"),
    ("incomplete tie diagnostic (no divergence_step)", {},
     dict(acc_kw=dict(ties=[("vt", 2)], tie_mut=lambda d: {k: v for k, v in d.items() if k != "divergence_step"})),
     "divergence_step missing"),
    ("acceptance inputs: only one unrelated manifest, no predictions", {}, dict(acc_kw=dict(layout_mut=only_unrelated)),
     "acceptance run dense/"),
    ("acceptance inputs: wrong data id in a run", {}, dict(acc_kw=dict(layout_mut=wrong_data)), "data id differs"),
    ("acceptance inputs: an order run missing", {}, dict(acc_kw=dict(layout_mut=drop_order_ba)), "order_ba_cobs__cobs+g2f"),
    ("acceptance report uids differ from the dense run", {}, dict(acc_kw=dict(report_mut=foreign_uid)),
     "same-input uids for niah_single_2 differ"),
    ("stale report: report changed after the acceptance receipt", {}, dict(after_receipts=rewrite_report),
     "acceptance receipt is for another report"),
    ("acceptance receipt missing", {}, dict(acc_kw=dict(receipt=False)), "acceptance receipt missing"),
    ("order_ba prediction differs (same uid), report still identical", {}, dict(acc_kw=dict(layout_mut=order_ba_differs)),
     "predictions differ across ab / ba / single"),
    ("order runs with zero coverage while the report claims 3", {}, dict(acc_kw=dict(layout_mut=no_coverage)),
     "contradicts the bound runs"),
    ("order runs with zero coverage and a report saying 0", {},
     dict(acc_kw=dict(layout_mut=no_coverage, report_mut=report_zero_cov)), "no row covered"),
    ("report covered_rows differ from the runs", {}, dict(acc_kw=dict(report_mut=report_cov_2)), "contradicts the bound runs"),
    ("input changed during the evaluation", {}, dict(acc_kw=dict(during_eval=CHANGE_DURING)),
     "evaluation provenance missing"),
    ("report existed before the evaluation", {}, dict(acc_kw=dict(pre_existing_report=True)), "evaluation provenance missing"),
    ("report replaced after its evaluation", {}, dict(acc_kw=dict(after_eval=rewrite_report)), "changed after its evaluation"),
    ("inputs changed after the receipt and accept-receipt re-run", {}, dict(after_receipts=change_and_rereceipt),
     "changed or missing since the receipt"),
    ("control: full watch, dense/vt prediction changed after the evaluation", {}, dict(acc_kw=dict(after_eval=change_dense_vt)),
     "differs from its evaluation snapshot"),
    ("report evaluation watched data only, dense/vt changed afterwards", {},
     dict(acc_kw=dict(watch_report=lambda pred, data: [data], after_eval=change_dense_vt)), "did not cover required input"),
    ("report evaluation watched data only (incomplete call, no change)", {},
     dict(acc_kw=dict(watch_report=lambda pred, data: [data])), "did not cover required input"),
    ("tie diagnostic evaluation watched data only", {},
     dict(acc_kw=dict(ties=[("vt", 2)], watch_diag=lambda pred, data: [data])), "did not cover required input"),
    ("verdict evaluation omitted the tie diagnostic", {},
     dict(acc_kw=dict(ties=[("vt", 2)], watch_verdict=lambda rp, diags: [rp])), "did not cover required input"),
    ("shard content swapped after the receipts (same size, same ns mtime)", {}, dict(after_receipts=swap_shard_content),
     "does not bind this probe"),
    ("acceptance identity after a same-size same-mtime content swap", {}, dict(after_receipts=swap_shard_content),
     "acceptance receipt model content differs"),
    ("verdict NOT ACCEPTED", {}, dict(acc_kw=dict(verdict="NOT ACCEPTED: x")), "does not match the decision"),
    ("acceptance for another model", {}, dict(acc_kw=dict(acc_code_mut=lambda c: c.__setitem__("model:config.json", "0" * 16))),
     "for another model/code"),
    ("acceptance for other code", {}, dict(acc_kw=dict(acc_code_mut=lambda c: c.__setitem__("sparse_attn.py", "0" * 16))),
     "for another model/code"),
    ("probe without phases", {}, dict(probe_kw=dict(probe_mut=lambda r: r.pop("phases"))), "phases missing"),
    ("probe phases only prefill + decode:dense", {}, dict(probe_kw=dict(probe_mut=phase_only_dense)), "phases missing"),
    ("probe phase peak 30 vs top-level 10 (capacity 22)", {}, dict(probe_kw=dict(probe_mut=inconsistent_peak)), "!= max over phases"),
    ("probe without 10% headroom", {}, dict(probe_kw=dict(peak=21.0)), "exceeds the device capacity"),
    ("probe non-finite measurement", {}, dict(probe_kw=dict(probe_mut=lambda r: r["phases"]["prefill"].__setitem__("peak_alloc_gib", -1))),
     "non-finite or negative"),
    ("probes missing a block", {}, dict(probe_kw=dict(blocks=(16, 32))), "tier needs"),
    ("probe receipt missing", {}, dict(probe_kw=dict(receipt=False)), "receipt"),
    ("probe receipt for older weights (stale identity)", {}, dict(probe_kw=dict(stale=True)), "does not bind this probe"),
)
for nm, mkw, rkw, why in REFUSALS:
    r0 = tempfile.mkdtemp(prefix="aggv2_", dir=os.environ.get("TMPDIR"))
    model_dir(r0, **mkw)
    build_data(r0)
    rc, out = register(r0, **rkw)
    check(rc != 0 and "REGISTER REFUSED" in out and why in out, f"register refuses: {nm}", out)
    shutil.rmtree(r0)


# ---- I1 / I3: aggregation refusals
def agg_case(nm, mutate_after_bind=None, rd_mut=None, unit_mut=None, man_mut=None, other_rd=False, must="GATE FAILED"):
    root, outs, dirs = full(unit_mut=None, man_mut=man_mut)
    bind_all(root, dirs)
    if rd_mut:
        p = os.path.join(root, "rd.json")
        j = json.load(open(p))
        rd_mut(j)
        json.dump(j, open(p, "w"))
    if unit_mut:
        p = os.path.join(root, "data", M, str(PC["length"]), "unit_lengths_b32_n200.json")
        j = json.load(open(p))
        unit_mut(j)
        json.dump(j, open(p, "w"))
    if mutate_after_bind:
        mutate_after_bind(root)
    rd = "rd.json"
    if other_rd:
        rc0, o0 = register(root, out="rd2.json", probe_kw=dict(peak=9.0))
        assert rc0 == 0, o0
        rd = "rd2.json"
    rc, out = aggregate(root, rd)
    check(rc == 2 and must in out, f"aggregation refuses: {nm}", out)
    shutil.rmtree(root)


def first_shard(j):
    return SHARDS[0]


agg_case("descriptor weight hash not a sha256", must="not a sha256",
         rd_mut=lambda j: j["model"]["files"][SHARDS[0]].__setitem__("sha256", "not-a-sha256"))
agg_case("descriptor missing a shard", must="not in the descriptor", rd_mut=lambda j: j["model"]["files"].pop(SHARDS[1]))
agg_case("descriptor of another revision", must="differs from the registered revision",
         rd_mut=lambda j: j["model"].__setitem__("revision", "f" * 40))
agg_case("manifest shard size/mtime differs", must="differs from the registered weights",
         man_mut=lambda m: m["code"].__setitem__(f"model:{SHARDS[0]}", "size=1 mtime=1"))
agg_case("legal class change in the unit file", must="unit file differs from the one frozen",
         unit_mut=lambda u: u["tasks"]["vt"]["samples"][1].__setitem__("cls", ["repair"]))
agg_case("common wrong code id in every manifest", must="id differs from the registered code",
         man_mut=lambda m: m["code"].__setitem__("sparse_attn.py", "wrong-source-id0"))


def drift(root):
    p = os.path.join(root, "pred", M, str(PC["length"]), f"budget{PC['budget']}", "b8", f"t{TIER}_n100", "quest@1024", "vt.jsonl")
    rows = [json.loads(x) for x in open(p)]
    rows[3]["pred"] = "changed"
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))


agg_case("prediction changed after binding", mutate_after_bind=drift, must="manifest/predictions changed since")
agg_case("bindings for another descriptor", other_rd=True, must="for another descriptor")
agg_case("classes on a no-unit task", must="no-unit task has unit lengths or classes",
         unit_mut=lambda u: u["tasks"]["cwe"]["samples"][0].__setitem__("cls", ["in0"]))

print("PROBLEMS", bad if bad else "none")
sys.exit(1 if bad else 0)
