"""Synthetic tests of src/phase2_gate.py against an independent spec. A well-formed partial file passes only with
--partial (and says INCOMPLETE); every listed defect fails with exit 2, including Codex F2 ①–⑤ cases (manifest and rows
consistently switched to another code identity; dev flag only on rows; missing identity fields / wrong config;
haystack start/n_chars/books wrong; strict not re-computable). Exit 1 on failure."""
import copy
import json
import os
import subprocess
import sys
import tempfile

ROOT = __file__.rsplit("/theory/", 1)[0]
GATE = [sys.executable, f"{ROOT}/src/phase2_gate.py"]
ARMS = ["cobs", "cobs+g2f", "nf_pt1", "nf_pt1+g2f", "quest", "quest+g2f"]
SLOT = 98304
ARGV = ["--ctx", "16384", "--haystack", "v2"]
CODE = {"sparse_attn.py": "a", "run_e2e.py": "b", "tasks.py": "c", "corpus_v2": "d", "model:config.json": "e"}
FIXED = dict(block=32, local=64, rank=8, w_per_head=False, prefill_chunk=2048, force=None, base_shift=0,
             budget=512, ctx=16384)
BOOKS = [["pg1.txt", 0], ["pg2.txt", 9_840_000], ["pg3.txt", 30_000_000]]
import hashlib  # noqa: E402
ANALYSIS = {f: hashlib.sha256(open(f"{ROOT}/{f}", "rb").read()).hexdigest() for f in
            ["src/phase2_gate.py", "src/phase2_analysis.py", "src/newcombe.py", "src/make_phase2_spec.py",
             "src/tasks.py", "theory/prereg_g2f_v3.md"]}
SPEC = dict(argv=ARGV, model="m", code=CODE, fixed=FIXED, methods=["dense"] + ARMS, analysis=ANALYSIS,
            keys=[["multikey", 210000 + i, 100 + 2 * i] for i in range(150)] +
                 [["single_essay", 220000 + i, 101 + 2 * i] for i in range(150)],
            slot_chars=SLOT, books=BOOKS, corpus_chars=42_000_000)
MAN = dict(argv=ARGV, methods=ARMS, model="m", code=CODE, dev_flags=[], legacy_rows_before_manifest=0)


def books_for(start, end):
    return [n for i, (n, b0) in enumerate(BOOKS) if b0 < end and (BOOKS[i + 1][1] if i + 1 < len(BOOKS) else 1e18) > start]


def row(task, seed, slot):
    tok = dict(min=1, max=1, mean=1, calls=10, budget_violations=0, max_abs_budget_error=0,
               fixed_region_violations=0, budget_checked_calls=10)
    res = {f"{m}@512": dict(out="The number is 1234567.", score=1.0, strict=1,
                            tokens=None if m == "dense" else dict(tok)) for m in ["dense"] + ARMS}
    start = slot * SLOT
    r = dict(task=task, ctx=16384, seed=seed, L=16380, answers=["1234567"], prompt_sha="p", token_sha="t",
             spans=[{"value": "1234567"}], res=res, run_id=dict(code=dict(CODE), argv=list(ARGV), model="m"),
             hay=dict(corpus="v2", slot=slot, start=start, n_chars=74000, books=books_for(start, start + 74000)))
    r.update({k: v for k, v in FIXED.items() if k not in ("ctx", "budget")})
    return r


def run(rows, man=MAN, partial=True, spec=SPEC, tool=None):
    d = tempfile.mkdtemp()
    sp, p = os.path.join(d, "spec.json"), os.path.join(d, "p2.jsonl")
    json.dump(spec, open(sp, "w"))
    open(p, "w").write("".join(json.dumps(r) + "\n" for r in rows))
    json.dump(man, open(p + ".manifest.json", "w"))
    cmd = (tool or GATE) + [sp, p] + (["--partial"] if partial and tool is None else [])
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.returncode, out.stdout


good = [row("multikey", 210000, 100), row("single_essay", 220000, 101)]
bad = []
rc, out = run(good)
if rc != 0 or "INCOMPLETE" not in out:
    bad.append(f"good partial file: rc {rc} {out[-200:]}")
if run(good, partial=False)[0] != 2:
    bad.append("incomplete file accepted without --partial")


def case(name, f, man=MAN):
    rs = copy.deepcopy(good)
    f(rs)
    if run(rs, man)[0] != 2:
        bad.append(f"{name} accepted")


other = dict(CODE, **{"sparse_attn.py": "zzz"})
for r in good:
    pass
case("manifest+rows switched to another code (F2-1)",
     lambda rs: [r["run_id"].__setitem__("code", dict(other)) for r in rs], man=dict(MAN, code=dict(other)))
case("dev flag only on rows (F2-2)", lambda rs: rs[0]["run_id"].__setitem__("argv", ARGV + ["--allow-code-change"]))
case("manifest legacy rows (F2-2)", lambda rs: None, man=dict(MAN, legacy_rows_before_manifest=3))
case("manifest code changes (F2-2)", lambda rs: None, man=dict(MAN, code_changes=[{"at": "x"}]))
case("missing prompt_sha (F2-3)", lambda rs: rs[0].pop("prompt_sha"))
case("missing spans (F2-3)", lambda rs: rs[1].__setitem__("spans", []))
case("wrong rank (F2-3)", lambda rs: rs[0].__setitem__("rank", 16))
case("missing base_shift (F2-3)", lambda rs: rs[0].pop("base_shift"))
case("strict not re-computable (F2-3)", lambda rs: rs[0]["res"]["cobs@512"].__setitem__("out", "no digits"))
case("start=0 with slot 100 (F2-4)", lambda rs: rs[0]["hay"].__setitem__("start", 0))
case("n_chars above slot (F2-4)", lambda rs: rs[0]["hay"].__setitem__("n_chars", SLOT + 1))
case("books not matching index (F2-4)", lambda rs: rs[0]["hay"].__setitem__("books", ["pg3.txt"]))
case("wrong slot", lambda rs: rs[0]["hay"].__setitem__("slot", 7))
case("missing counter", lambda rs: rs[0]["res"]["cobs@512"]["tokens"].pop("budget_checked_calls"))
case("unchecked calls", lambda rs: rs[0]["res"]["cobs@512"]["tokens"].__setitem__("budget_checked_calls", 9))
case("budget violation", lambda rs: rs[0]["res"]["quest@512"]["tokens"].__setitem__("budget_violations", 1))
case("extra method", lambda rs: rs[0]["res"].__setitem__("em@512", rs[0]["res"]["cobs@512"]))
case("duplicate key", lambda rs: rs.append(copy.deepcopy(rs[0])))
# Codex C3: a changed frozen analysis file must fail
if run(good, spec=dict(SPEC, analysis=dict(ANALYSIS, **{"src/tasks.py": "0" * 64})))[0] != 2:
    bad.append("changed frozen analysis file accepted")
if run(good, spec={k: v for k, v in SPEC.items() if k != "analysis"})[0] != 2:
    bad.append("SPEC without analysis hashes accepted")
# full synthetic 300-key file: gate passes in full mode and the analysis entry runs; partial file is refused by it
full = [row(t, sd, sl) for t, sd, sl in SPEC["keys"]]
for i, r in enumerate(full):                       # make cobs wrong on some prompts so the contrast is non-trivial
    if i % 10 == 0:
        r["res"]["cobs@512"]["out"], r["res"]["cobs@512"]["strict"] = "no", 0
rc, out = run(full, partial=False)
if rc != 0 or "PHASE-2 GATE PASSED" not in out:
    bad.append(f"full synthetic file rejected: {out[-300:]}")
ANA = [sys.executable, f"{ROOT}/src/phase2_analysis.py"]
rc, out = run(full, partial=False, tool=ANA)
if rc != 0 or '"primary"' not in out or '"cells": [\n   270,\n   30' not in out:
    bad.append(f"analysis on full file: rc {rc} {out[:300]}")
# Codex R1: containment is reported separately from strict (target 1234567, output 81234567 -> containment 1, strict 0)
full2 = copy.deepcopy(full)
for r in full2:
    r["res"]["quest@512"]["out"], r["res"]["quest@512"]["strict"] = "It is 81234567.", 0
rc, out = run(full2, partial=False, tool=ANA)
try:
    res = json.loads(out)
    q = res["containment_descriptive"]["per_arm"]["quest"]
    qs = res["secondary"]["quest"]["acc0"]
    if not (q["count"] == 300 and qs == 0.0):
        bad.append(f"containment/strict conflated: containment {q}, strict quest acc {qs}")
except Exception as e:  # noqa: BLE001
    bad.append(f"analysis output not parseable for R1 check: {e} {out[:200]}")
if run(good, partial=False, tool=ANA)[0] != 2:
    bad.append("analysis ran on an incomplete file")
p1 = f"{ROOT}/runs/phase1_g2f_dev.jsonl"
d = tempfile.mkdtemp()
json.dump(SPEC, open(os.path.join(d, "spec.json"), "w"))
if subprocess.run(GATE + [os.path.join(d, "spec.json"), p1, "--partial"], capture_output=True).returncode != 2:
    bad.append("phase-1 file accepted")
print("PROBLEMS", bad if bad else "none")
sys.exit(1 if bad else 0)
