"""Targeted checks for src/kv_prediction_confirm.py (Codex K1-a/K1-b), without confirmation data:
  1. receipt binding: the real em-dev receipt (another job) is refused; a receipt with the right id/label/status but a
     changed argv is refused; one whose SPEC fingerprint differs is refused; a receipt assembled to match the SPEC
     exactly yields no problems (synthetic positive control for the binding logic only);
  2. empty pooled group: --dev on a development subset containing only kv-inside prompts prints "insufficient
     information" and does not call the registered script (no division by zero), exit 0.
Exit 1 on failure."""
import copy
import json
import os
import subprocess
import sys
import tempfile

ROOT = __file__.rsplit("/theory/", 1)[0]
sys.path.insert(0, ROOT + "/src")
import kv_prediction_confirm as K  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

bad = []
spec_path = ROOT + "/theory/phase2_spec.json"
spec = json.load(open(spec_path))
run = spec["argv"][spec["argv"].index("--out") + 1]
em = json.load(open("/home/kooki3/.local/state/work-handoff/runs/9eebd362ffd5417c8050690e2aaeb529/receipt.json"))
if not K.receipt_problems(em, spec, spec_path, run):
    bad.append("another job's receipt accepted")
good = dict(id=K.CONFIRM_JOB, project="dual-channel-attention", label="phase2-confirm", phase="finished",
            status="succeeded", exit_code=0,
            command=["taskset", "-c", "0-5", "python", os.path.join(ROOT, "src", "run_e2e.py")] + spec["argv"],
            inputs=[dict(path=os.path.abspath(spec_path), fingerprint=dict(sha256=K.sha(spec_path)))])
if K.receipt_problems(good, spec, spec_path, run):
    bad.append(f"matching receipt refused: {K.receipt_problems(good, spec, spec_path, run)}")
r2 = copy.deepcopy(good); r2["command"] = r2["command"][:-1] + ["/tmp/other.jsonl"]
if not K.receipt_problems(r2, spec, spec_path, run):
    bad.append("changed argv accepted")
r3 = copy.deepcopy(good); r3["inputs"][0]["fingerprint"]["sha256"] = "0" * 64
if not K.receipt_problems(r3, spec, spec_path, run):
    bad.append("wrong SPEC fingerprint accepted")
r4 = copy.deepcopy(good); r4["status"], r4["exit_code"] = "failed", 1
if not K.receipt_problems(r4, spec, spec_path, run):
    bad.append("failed status accepted")
# 2. empty pooled group
tok = AutoTokenizer.from_pretrained(K.TOKENIZER)
rows = [json.loads(l) for l in open(ROOT + "/runs/phase1_g2f_dev.jsonl")]
_, tab = K.token_check(rows, tok)
inside = [r for r, t in zip(rows, tab) if not t["cross"]][:6]
d = tempfile.mkdtemp()
p = os.path.join(d, "inside_only.jsonl")
open(p, "w").write("".join(json.dumps(r) + "\n" for r in inside))
out = subprocess.run([sys.executable, ROOT + "/src/kv_prediction_confirm.py", "--dev", p], capture_output=True, text=True)
if out.returncode != 0 or "insufficient information" not in out.stdout or "registered P-kv (original" in out.stdout:
    bad.append(f"empty pooled group branch: rc {out.returncode} {out.stdout[-300:]} {out.stderr[-300:]}")
print("PROBLEMS", bad if bad else "none")
sys.exit(1 if bad else 0)
