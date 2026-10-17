"""End-to-end check of cluster/aggregate_appendix.py on synthetic data (official RULER metric functions from the pinned
checkout; no model): appendix group "sum" (mean, mean+g2f, quest) and group "delim" (cobs, cobs+g2f, delim_cobs) in
their appendix run tags. Expect: mean+g2f vs quest32 and delim_cobs vs cobs+g2f pass the frozen gate and are
DESCRIPTIVE with no p; cobs+g2f vs cobs in the appendix tag is DESCRIPTIVE (never PRIMARY); a non-appendix run tag is
refused; plan_v1.json on disk is unchanged. Exit 1 on failure."""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = __file__.rsplit("/theory/", 1)[0]
PLAN_PATH = ROOT + "/cluster/plan_v1.json"
PLAN = json.load(open(PLAN_PATH))
RULER = "/home/kooki3/tools/sources/ruler"
PY = sys.executable
N = 100
PC = PLAN["primary_cell"]
CODE = {"sparse_attn.py": "a", "ruler_pred.py": "b", "model:config.json": "c", "model:tokenizer.json": "d",
        "model:tokenizer_config.json": "e", "model:generation_config.json": "f", "model:revision": "rev123"}
_s = importlib.util.spec_from_file_location("tc", os.path.join(RULER, "scripts", "data", "synthetic", "constants.py"))
TC = importlib.util.module_from_spec(_s)
_s.loader.exec_module(TC)
import yaml  # noqa: E402
CFG = yaml.safe_load(open(os.path.join(RULER, "scripts", "synthetic.yaml")))
GROUPS = {"sum": ("mean,mean+g2f,quest", {"mean": 30, "mean+g2f": 10, "quest": 0}),
          "delim": ("cobs,cobs+g2f,delim_cobs", {"cobs": 30, "cobs+g2f": 0, "delim_cobs": 20})}


def build(root):
    data = os.path.join(root, "data", PC["model"], str(PC["length"]))
    outs = {}
    for t in PLAN["tasks"]:
        os.makedirs(os.path.join(data, t), exist_ok=True)
        f = os.path.join(data, t, "validation.jsonl")
        open(f, "w").write("".join(json.dumps(dict(index=i, input=f"ctx {t} {i}", answer_prefix=" Answer:",
                                                   outputs=["alpha"], length=10)) + "\n" for i in range(N)))
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        outs[t] = dict(sha256=h, sample_uids=[f"{h[:16]}:{t}:{i}" for i in range(N)])
    json.dump(dict(ruler_commit=PLAN["ruler_commit"], outputs=outs), open(os.path.join(data, "prep_manifest.json"), "w"))
    tok = dict(calls=5, budget_checked_calls=5, budget_violations=0, max_abs_budget_error=0, fixed_region_violations=0)
    for g, (methods, wrong) in GROUPS.items():
        d = os.path.join(root, "pred", PC["model"], str(PC["length"]), f"budget{PC['budget']}", "b32",
                         f"t3_n{N}_appendix-{g}")
        for m, w in wrong.items():
            os.makedirs(os.path.join(d, f"{m}@{PC['budget']}"), exist_ok=True)
            for t in PLAN["tasks"]:
                rows = [dict(sample_uid=u, input_sha=hashlib.sha256(f"ctx {t} {i} Answer:".encode()).hexdigest()[:16],
                             token_sha=f"tok{i}", outputs=["alpha"], pred="no idea" if i < w else "it is alpha",
                             tokens=dict(tok)) for i, u in enumerate(outs[t]["sample_uids"])]
                open(os.path.join(d, f"{m}@{PC['budget']}", f"{t}.jsonl"), "w").write(
                    "".join(json.dumps(r) + "\n" for r in rows))
        for t in PLAN["tasks"]:
            json.dump(dict(code=dict(CODE, data=outs[t]["sha256"][:16]), ruler_commit=PLAN["ruler_commit"],
                           tokens_to_generate=TC.TASKS[CFG[t]["task"]]["tokens_to_generate"], stop_ids=[1, 2],
                           special_tokens=True, argv=["--methods", methods, "--budget", str(PC["budget"]), "--block", "32",
                                                      "--limit", str(N), "--strict-budget"]),
                      open(os.path.join(d, f"manifest.{t}.json"), "w"))


def agg(root, a, b, tag):
    r = subprocess.run([PY, ROOT + "/cluster/aggregate_appendix.py", "--ruler", RULER, "--out-root", root, "--model",
                        PC["model"], "--length", str(PC["length"]), "--budget", str(PC["budget"]), "--n", str(N),
                        "--arm-a", a, "--arm-b", b, "--run-tag", tag], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


plan_before = hashlib.sha256(open(PLAN_PATH, "rb").read()).hexdigest()
bad = []
root = tempfile.mkdtemp()
build(root)
for a, b, g, D in (("mean+g2f", "quest32", "sum", -10.0), ("delim_cobs", "cobs+g2f", "delim", -20.0),
                   ("cobs+g2f", "cobs", "delim", 30.0)):
    rc, out = agg(root, a, b, f"t3_n{N}_appendix-{g}")
    ok = rc == 0 and "[DESCRIPTIVE]" in out and "no test (descriptive)" in out and f"D = {D:+.3f}" in out
    print(f"{a} vs {b} ({g}): rc {rc}, DESCRIPTIVE {'[DESCRIPTIVE]' in out}, D as expected {f'D = {D:+.3f}' in out}")
    if not ok:
        bad.append(f"{a} vs {b}: {out[-400:]}")
rc, out = agg(root, "mean+g2f", "quest32", f"t3_n{N}")
if rc == 0:
    bad.append("non-appendix run tag accepted")
print(f"non-appendix tag refused: {rc != 0}")
if hashlib.sha256(open(PLAN_PATH, "rb").read()).hexdigest() != plan_before:
    bad.append("plan_v1.json changed on disk")
shutil.rmtree(root)
print("PROBLEMS", bad if bad else "none")
sys.exit(1 if bad else 0)
