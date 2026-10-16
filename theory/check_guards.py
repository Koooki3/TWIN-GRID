"""V2: resume guard refuses incompatible outputs BEFORE model load. Runs run_e2e.py as a subprocess on
synthetic output files (refusal paths exit before any model/GPU use; run with CUDA_VISIBLE_DEVICES="").
Each case must be refused for its specific reason; a compatible control row must pass validation.
Exit 1 on any unexpected outcome."""
import json
import os
import subprocess
import sys
import tempfile

PY = sys.executable
RUN = os.path.join(sys.argv[1] if len(sys.argv) > 1 else "src", "run_e2e.py")
BASE = ["--ctx", "16384", "--budgets", "512", "--tasks", "multikey", "--n", "1", "--methods", "cobs,quest"]
MODEL = "/home/kooki3/models/qwen2.5-1.5b-instruct"
CODE = {"a": "0"}


def full_row(seed, argv, code=CODE, methods=("cobs", "quest")):
    res = {f"{m}@512": dict(out=" 1", score=1.0, strict=1.0, tokens={}) for m in methods}
    res["dense@512"] = dict(out=" 1", score=1.0, strict=1.0, tokens={})
    return dict(task="multikey", ctx=16384, seed=seed, res=res, token_sha="x", spans=[],
                run_id=dict(argv=argv, code=code, model=MODEL))


def run_case(make_lines, manifest=True, manifest_argv=None):
    d = tempfile.mkdtemp(prefix="guard_", dir=os.environ.get("TMPDIR", "/tmp"))
    out = os.path.join(d, "o.jsonl")
    argv = BASE + ["--out", out]
    open(out, "w").write(make_lines(argv))
    if manifest:
        json.dump(dict(argv=manifest_argv or argv, methods=["cobs", "quest"], model=MODEL, code=CODE),
                  open(out + ".manifest.json", "w"))
    p = subprocess.run([PY, RUN] + argv + ["--allow-code-change", "--validate-only"], capture_output=True, text=True,
                       timeout=300)
    text = p.stderr + p.stdout
    return p.returncode, text


def line(r):
    return json.dumps(r) + "\n"


cases = [
    ("no manifest, rows present", lambda av: line(full_row(1000, av)), False, None, "but no manifest"),
    ("truncated tail", lambda av: line(full_row(1000, av)) + '{"task": "multi', True, None, "not valid JSON"),
    ("duplicate row", lambda av: line(full_row(1000, av)) * 2, True, None, "duplicate row"),
    ("different method set", lambda av: line(full_row(1000, av, methods=("cobs",))), True, None,
     "different run identity"),
    ("different code identity", lambda av: line(full_row(1000, av, code={"a": "1"})), True, None,
     "different run identity"),
    ("missing required fields", lambda av: line({**full_row(1000, av), "token_sha": None} if False else
                                                 {k: v for k, v in full_row(1000, av).items() if k != "token_sha"}),
     True, None, "lacks required fields"),
    ("argv differs", lambda av: line(full_row(1000, av)), True, ["--ctx", "8192"], "argv/methods/model differ"),
]
ok = True
for name, mk, man, man_argv, expect in cases:
    rc, text = run_case(mk, man, man_argv)
    hit = rc != 0 and expect in text
    print(f"{name:26s} rc={rc} refused_for_expected_reason={hit}")
    ok &= hit
rc, text = run_case(lambda av: line(full_row(1000, av)))
passed = rc == 0 and "VALIDATION PASSED" in text and "1 existing rows accepted" in text
print(f"{'compatible row (control)':26s} rc={rc} passed_validation={passed} (positive sentinel, no model load)")
ok &= passed
print("ALL OK" if ok else "PROBLEM")
sys.exit(0 if ok else 1)
