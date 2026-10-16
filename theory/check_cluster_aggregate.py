"""End-to-end checks of cluster/aggregate.py v3 and cluster/holm.py v2 on synthetic data (official RULER metric
functions from the real pinned checkout; no model). Exit 1 on failure.
Positive: primary cell / pair -> PRIMARY with D = 100*K/N exactly and small p; both secondary members (v1.4: one on
the registered coverable subset from a synthetic unit-length file) -> SECONDARY, subset = the 7 unit tasks;
cross-block pair -> DESCRIPTIVE, no p; Holm: complete family of two consistent results -> computed (exit 0);
incomplete family, duplicate member, mixed data identity, invalid p, members without identity -> exit 3; subset
without a unit file, a unit file of another tokenizer or bound to other data -> exit 2.
Negatives (each exit 2): common wrong answers, common wrong input, data drift, n > file, wrong manifest budget, wrong
prep RULER commit, missing uid, budget violation, duplicate uid; and (Codex Q2) a violation counter key deleted,
calls given as bool, both arms missing the same identity key, unrecorded model revision, a different --local / a
--no-special-tokens / different stop_ids in one arm, wrong tokens_to_generate, a changed token_sha in one arm, a
truncated or replaced prep uid list, and a modified RULER metric source under the same HEAD (copied checkout)."""
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = __file__.rsplit("/theory/", 1)[0]
PLAN = json.load(open(ROOT + "/cluster/plan_v1.json"))
RULER = "/home/kooki3/tools/sources/ruler"
PY = sys.executable
N, K = 100, 30
PC = PLAN["primary_cell"]
CODE = {"sparse_attn.py": "a", "ruler_pred.py": "b", "model:config.json": "c", "model:tokenizer.json": "d",
        "model:tokenizer_config.json": "e", "model:generation_config.json": "f", "model:revision": "rev123"}
sys.path.insert(0, os.path.join(RULER, "scripts", "data", "synthetic"))
import importlib.util  # noqa: E402
_s = importlib.util.spec_from_file_location("tc", os.path.join(RULER, "scripts", "data", "synthetic", "constants.py"))
TC = importlib.util.module_from_spec(_s)
_s.loader.exec_module(TC)
import yaml  # noqa: E402
CFG = yaml.safe_load(open(os.path.join(RULER, "scripts", "synthetic.yaml")))
ARMS4 = ("quest16+g2f", "quest16", "cobs+g2f", "cobs", "em", "quest32", "cobs+ptr", "nf_pt1", "nf_pt1+g2f")
NOT_COV, NO_UNIT = ("niah_single_3", "niah_multikey_3"), ("cwe", "fwe", "qa_1", "qa_2")


def build(root, mutate=None, prep_mutate=None, prep_commit=None):
    data = os.path.join(root, "data", PC["model"], str(PC["length"]))
    outs = {}
    for t in PLAN["tasks"]:
        os.makedirs(os.path.join(data, t), exist_ok=True)
        f = os.path.join(data, t, "validation.jsonl")
        lines = [dict(index=i, input=f"ctx {t} {i}", answer_prefix=" Answer:", outputs=["alpha"], length=10)
                 for i in range(N)]
        open(f, "w").write("".join(json.dumps(x) + "\n" for x in lines))
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        outs[t] = dict(sha256=h, sample_uids=[f"{h[:16]}:{t}:{i}" for i in range(N)])
    if prep_mutate:
        prep_mutate(outs)
    json.dump(dict(ruler_commit=prep_commit or PLAN["ruler_commit"], outputs=outs),
              open(os.path.join(data, "prep_manifest.json"), "w"))
    tok = dict(calls=5, budget_checked_calls=5, budget_violations=0, max_abs_budget_error=0, fixed_region_violations=0)
    groups = {16: "quest,quest+g2f", 32: "cobs,cobs+g2f,em,quest,cobs+ptr,nf_pt1,nf_pt1+g2f"}
    for arm_name in ARMS4:
        arm = PLAN["arms"][arm_name]
        wrong = 0 if arm_name.endswith("+g2f") else K
        d = os.path.join(root, "pred", PC["model"], str(PC["length"]), f"budget{PC['budget']}", f"b{arm['block']}",
                         f"t2_n{N}")
        os.makedirs(os.path.join(d, f"{arm['method']}@{PC['budget']}"), exist_ok=True)
        for t in PLAN["tasks"]:
            man = dict(code=dict(CODE, data=outs[t]["sha256"][:16]), ruler_commit=PLAN["ruler_commit"],
                       tokens_to_generate=TC.TASKS[CFG[t]["task"]]["tokens_to_generate"], stop_ids=[1, 2],
                       special_tokens=True,
                       argv=["--methods", groups[arm["block"]], "--budget", str(PC["budget"]), "--block",
                             str(arm["block"]), "--limit", str(N), "--strict-budget"])
            rows = []
            for i, u in enumerate(outs[t]["sample_uids"]):
                inp = f"ctx {t} {i}" + " Answer:"
                rows.append(dict(sample_uid=u, input_sha=hashlib.sha256(inp.encode()).hexdigest()[:16],
                                 token_sha=f"tok{i}", outputs=["alpha"],
                                 pred="it is alpha" if i >= wrong else "no idea", tokens=dict(tok)))
            if mutate:
                mutate(arm_name, t, rows, man, data)
            mp = os.path.join(d, f"manifest.{t}.json")
            if not os.path.exists(mp) or mutate:
                json.dump(man, open(mp, "w"))
            open(os.path.join(d, f"{arm['method']}@{PC['budget']}", f"{t}.jsonl"), "w").write(
                "".join(json.dumps(r) + "\n" for r in rows))


def write_units(root, tok="d" + "0" * 63, drift=None):
    """Synthetic unit-length file bound to the synthetic data (coverable: 7 tasks; 2 not coverable; 4 no unit)."""
    data = os.path.join(root, "data", PC["model"], str(PC["length"]))
    prep = json.load(open(os.path.join(data, "prep_manifest.json")))["outputs"]
    tasks = {}
    for t in PLAN["tasks"]:
        lens = [] if t in NO_UNIT else ([40] if t in NOT_COV else [10])
        tasks[t] = dict(data_sha256=("0" * 64 if t == drift else prep[t]["sha256"]),
                        samples=[dict(lens=lens) for _ in range(N)])
    p = os.path.join(root, "units.json")
    json.dump(dict(version=PLAN["coverage_rule"]["version"], block=32, unit=32, local=64, tokenizer_json_sha256=tok,
                   tasks=tasks), open(p, "w"))
    return p


def agg(root, a, b, n=N, js=None, ruler=RULER, extra=()):
    cmd = [PY, ROOT + "/cluster/aggregate.py", "--ruler", ruler, "--out-root", root, "--model", PC["model"],
           "--length", str(PC["length"]), "--budget", str(PC["budget"]), "--n", str(n), "--arm-a", a, "--arm-b", b,
           "--run-tag", f"t2_n{N}"] + (["--json", js] if js else []) + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def holm(*paths):
    return subprocess.run([PY, ROOT + "/cluster/holm.py", *paths], capture_output=True, text=True)


bad = []
root = tempfile.mkdtemp()
build(root)
J = {k: os.path.join(root, f"{k}.json") for k in ("primary", "s1", "s2")}
UF = write_units(root)
rc, out = agg(root, *PLAN["primary_pair"], js=J["primary"])
m = re.search(r"macro difference D = ([+-][\d.]+)", out)
if rc != 0 or "[PRIMARY]" not in out or not m or abs(float(m.group(1)) - 100 * K / N) > 1e-9 or "p = 0.0000" not in out:
    bad.append(f"primary case: rc {rc} {out[-500:]}")
print(out.strip().splitlines()[-1])
for key, mem in zip(("s1", "s2"), PLAN["secondary_family"]):
    extra = ["--units", UF] + (["--subset", mem[2]] if len(mem) > 2 else [])
    rc, out = agg(root, mem[0], mem[1], js=J[key], extra=extra)
    if rc != 0 or "[SECONDARY]" not in out:
        bad.append(f"secondary {mem}: rc {rc} {out[-300:]}")
    if len(mem) > 2 and json.load(open(J[key])).get("subset_tasks") != sorted(
            t for t in PLAN["tasks"] if t not in NOT_COV + NO_UNIT):
        bad.append(f"subset tasks wrong for {mem}")
for name, extra in {"subset without units": ["--subset", "coverable"],
                    "units with another tokenizer": ["--units", write_units(root, tok="e" * 64), "--subset", "coverable"],
                    "units bound to other data": ["--units", write_units(root, drift="vt"), "--subset", "coverable"]}.items():
    if agg(root, "cobs+g2f", "cobs", extra=extra)[0] != 2:
        bad.append(f"{name} accepted")
UF = write_units(root)
rc, out = agg(root, "quest16+g2f", "cobs")
if rc != 0 or "[DESCRIPTIVE]" not in out or "no test" not in out:
    bad.append(f"cross-block case: rc {rc} {out[-300:]}")
h = holm(J["s1"], J["s2"])
if h.returncode != 0 or "threshold" not in h.stdout:
    bad.append(f"holm complete family: rc {h.returncode} {h.stdout[-300:]}")
print(h.stdout.strip())
if holm(J["s1"]).returncode != 3:
    bad.append("holm incomplete family accepted")
if holm(J["s1"], J["s1"], J["s2"]).returncode != 3:
    bad.append("holm duplicate member accepted")
mixed = json.load(open(J["s2"]))
mixed["data_sha256"]["vt"] = "0" * 64
json.dump(mixed, open(os.path.join(root, "mixed.json"), "w"))
if holm(J["s1"], os.path.join(root, "mixed.json")).returncode != 3:
    bad.append("holm mixed data identity accepted")
badp = json.load(open(J["s2"]))
badp["p"] = float("nan")
json.dump(badp, open(os.path.join(root, "badp.json"), "w"))
if holm(J["s1"], os.path.join(root, "badp.json")).returncode != 3:
    bad.append("holm invalid p accepted")
for k in ("s1", "s2"):                             # reviewer T6: all members stripped of identity -> refuse
    stripped = {kk: v for kk, v in json.load(open(J[k])).items() if kk not in ("identity", "data_sha256")}
    json.dump(stripped, open(os.path.join(root, f"strip_{k}.json"), "w"))
if holm(*[os.path.join(root, f"strip_{k}.json") for k in ("s1", "s2")]).returncode != 3:
    bad.append("holm accepted members without identity/data fingerprints")
rc, out = agg(root, "quest16+g2f", "quest16", n=N + 1)
if rc != 2:
    bad.append(f"n larger than file accepted: rc {rc}")
shutil.rmtree(root)


def only(arm_name, task):
    def deco(f):
        def g(a, t, rows, man, data):
            if a == arm_name and t == task:
                f(rows, man, data)
        return g
    return deco


def both(task):
    def deco(f):
        def g(a, t, rows, man, data):
            if a in ("quest16+g2f", "quest16") and t == task:
                f(rows, man, data)
        return g
    return deco


cases = {
    "common wrong answers": both("vt")(lambda rows, man, data: [r.__setitem__("outputs", ["beta"]) for r in rows]),
    "common wrong input": both("cwe")(lambda rows, man, data: [r.__setitem__("input_sha", "0" * 16) for r in rows]),
    "data drift": only("quest16", "qa_1")(lambda rows, man, data: open(os.path.join(data, "qa_1", "validation.jsonl"), "a").write("\n")),
    "wrong budget in manifest": only("quest16", "fwe")(lambda rows, man, data: man["argv"].__setitem__(man["argv"].index("--budget") + 1, "2048")),
    "missing uid": only("quest16", "qa_2")(lambda rows, man, data: rows.pop()),
    "budget violation": only("quest16", "niah_single_1")(lambda rows, man, data: rows[0]["tokens"].__setitem__("budget_violations", 1)),
    "duplicate uid": only("quest16+g2f", "niah_single_3")(lambda rows, man, data: rows.append(copy.deepcopy(rows[0]))),
    "violation key deleted": only("quest16", "niah_multikey_1")(lambda rows, man, data: rows[0]["tokens"].pop("fixed_region_violations")),
    "calls as bool": only("quest16", "niah_multikey_2")(lambda rows, man, data: rows[0]["tokens"].update(calls=True, budget_checked_calls=True)),
    "both arms missing identity key": both("niah_multikey_3")(lambda rows, man, data: man["code"].pop("sparse_attn.py")),
    "revision unrecorded": both("niah_multivalue")(lambda rows, man, data: man["code"].__setitem__("model:revision", "unrecorded")),
    "different --local": only("quest16", "niah_multiquery")(lambda rows, man, data: man["argv"].extend(["--local", "16"])),
    "no special tokens": only("quest16", "vt")(lambda rows, man, data: (man["argv"].append("--no-special-tokens"), man.__setitem__("special_tokens", False))),
    "different stop_ids": only("quest16", "cwe")(lambda rows, man, data: man.__setitem__("stop_ids", [9])),
    "wrong tokens_to_generate": both("fwe")(lambda rows, man, data: man.__setitem__("tokens_to_generate", 7)),
    "token_sha differs": only("quest16", "qa_1")(lambda rows, man, data: rows[3].__setitem__("token_sha", "other")),
}
for name, fn in cases.items():
    r = tempfile.mkdtemp()
    build(r, fn)
    rc, out = agg(r, "quest16+g2f", "quest16")
    if rc != 2:
        bad.append(f"{name}: rc {rc}")
    shutil.rmtree(r)
for name, pm in {"truncated prep uid list": lambda o: o["vt"]["sample_uids"].pop(),
                 "replaced prep uid": lambda o: o["cwe"]["sample_uids"].__setitem__(5, "x:cwe:5")}.items():
    r = tempfile.mkdtemp()
    build(r, prep_mutate=pm)
    rc, out = agg(r, "quest16+g2f", "quest16")
    if rc != 2:
        bad.append(f"{name}: rc {rc}")
    shutil.rmtree(r)
r = tempfile.mkdtemp()
build(r, prep_commit="0" * 40)
if agg(r, "quest16+g2f", "quest16")[0] != 2:
    bad.append("wrong RULER commit in prep accepted")
shutil.rmtree(r)
# modified metric source under the same HEAD: a copy of the checkout (with .git) whose constants.py is edited
r = tempfile.mkdtemp()
build(r)
rc_copy = os.path.join(r, "ruler_copy")
shutil.copytree(RULER, rc_copy, ignore=shutil.ignore_patterns("*.json", "*.jsonl"))
open(os.path.join(rc_copy, "scripts", "eval", "synthetic", "constants.py"), "a").write("\n# edited\n")
rc, out = agg(r, "quest16+g2f", "quest16", ruler=rc_copy)
if rc != 2 or "pinned sha256" not in out:
    bad.append(f"modified metric source accepted: rc {rc} {out[-200:]}")
shutil.rmtree(r)
# multi-reference rounding (Codex A5 example)
sys.path.insert(0, ROOT + "/cluster")
import aggregate  # noqa: E402
spec = importlib.util.spec_from_file_location("c", os.path.join(RULER, "scripts", "eval", "synthetic", "constants.py"))
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)
preds, refs = ["has a only", "nothing"], [["a", "b", "c"], ["a", "b", "c"]]
un = [aggregate.unrounded("string_match_all", p_, r_) for p_, r_ in zip(preds, refs)]
off = cm.string_match_all(preds, refs)
rounded_mean = sum(cm.string_match_all([p_], [r_]) for p_, r_ in zip(preds, refs)) / 2
print(f"multi-ref: unrounded mean {sum(un) / 2:.4f}, official {off}, mean of per-sample rounded {rounded_mean}")
if not (abs(sum(un) / 2 - off) <= 0.005 and off == 16.67 and abs(rounded_mean - 16.665) < 1e-9):
    bad.append("multi-reference rounding behaviour")
print("PROBLEMS", bad if bad else "none")
sys.exit(1 if bad else 0)
