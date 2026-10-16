"""Phase-2 (confirmation) integrity gate for prereg v3. Exit 2 on any failure; exit 0 only when every check passes.
Everything is compared with an INDEPENDENT frozen spec (make_phase2_spec.py), never only with the run's own manifest:
  - manifest: argv, model, code/corpus/model identities exactly equal to the spec; no dev flags, no legacy rows, no
    code changes;
  - rows: exactly the spec's keys (no duplicates, no extras); run_id code/argv/model equal to the spec (rows carrying a
    dev flag fail, nothing is stripped); required identity fields present (prompt_sha, token_sha, L, answers, spans);
    fixed configuration equal to the spec; haystack v2 with slot as in the spec, start = slot * slot_chars,
    0 < n_chars <= slot_chars, interval inside the corpus, books exactly those of the spec's index that overlap it,
    slots pairwise distinct; methods exactly the spec's at the budget; out/score/strict present and strict
    re-computed from out with the frozen scorer; every sparse arm: integer calls > 0, all checked, zero violations.
Missing keys: in full mode a failure. --partial reports progress and validates present rows; its success line says
INCOMPLETE and is never an analysis licence.
Usage: phase2_gate.py SPEC.json PHASE2.jsonl [--partial]"""
import hashlib
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tasks  # noqa: E402
from tasks import score_strict  # noqa: E402

DEV_FLAGS = {"--allow-code-change", "--adopt-legacy", "--validate-only"}
COUNTERS = ("calls", "budget_checked_calls", "budget_violations", "max_abs_budget_error", "fixed_region_violations")


def overlapping_books(spec, start, end):
    b = spec["books"]
    return [n for i, (n, b0) in enumerate(b) if b0 < end and (b[i + 1][1] if i + 1 < len(b) else float("inf")) > start]


def frozen_files(spec):
    """Codex C3: the analysis code actually loaded must be the frozen one (full SHA-256 in SPEC['analysis'])."""
    root = os.path.dirname(HERE)
    bad = []
    if os.path.abspath(tasks.__file__) != os.path.join(HERE, "tasks.py"):
        bad.append(f"tasks loaded from {tasks.__file__}")
    want = spec.get("analysis")
    if not want:
        return ["SPEC has no analysis-file hashes"]
    for f, h in want.items():
        got = hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest()
        if got != h:
            bad.append(f"frozen file changed: {f}")
    return bad


def check(spec, path, partial):
    bad = frozen_files(spec)
    man_path = path + ".manifest.json"
    man = json.load(open(man_path)) if os.path.exists(man_path) else None
    if man is None:
        bad.append("no manifest")
    else:
        if man.get("argv") != spec["argv"] or man.get("model") != spec["model"] or man.get("code") != spec["code"]:
            bad.append("manifest identity differs from the frozen spec")
        if man.get("dev_flags") or DEV_FLAGS & set(man.get("argv", [])):
            bad.append("manifest has dev flags")
        if man.get("legacy_rows_before_manifest") or man.get("code_changes"):
            bad.append("manifest records legacy rows or code changes")
    rows = [json.loads(l) for l in open(path)]
    slot_of = {(t, s): sl for t, s, sl in spec["keys"]}
    want = {(t, spec["fixed"]["ctx"], s) for t, s, _ in spec["keys"]}
    keys = Counter((r.get("task"), r.get("ctx"), r.get("seed")) for r in rows)
    bad += [f"duplicate key {k}" for k, v in keys.items() if v > 1]
    bad += [f"unexpected key {k}" for k in keys if k not in want]
    missing = want - set(keys)
    if missing and not partial:
        bad.append(f"{len(missing)} pre-registered keys missing")
    want_methods = {f"{m}@{spec['fixed']['budget']}" for m in spec["methods"]}
    L_slot = spec["slot_chars"]
    slots = Counter()
    for r in rows:
        tag = f"{r.get('task')}:{r.get('seed')}"
        rid = r.get("run_id") or {}
        if rid.get("code") != spec["code"] or rid.get("argv") != spec["argv"] or rid.get("model") != spec["model"]:
            bad.append(f"{tag}: run identity differs from the frozen spec")
        for f in ("prompt_sha", "token_sha", "L", "answers", "spans"):
            if r.get(f) in (None, "", []):
                bad.append(f"{tag}: missing {f}")
        for f, v in spec["fixed"].items():
            if f in ("ctx", "budget"):
                continue
            if f not in r or r[f] != v:
                bad.append(f"{tag}: {f}={r.get(f, '<missing>')} (frozen {v})")
        h = r.get("hay") or {}
        sl = slot_of.get((r.get("task"), r.get("seed")))
        st, nch = h.get("start"), h.get("n_chars")
        if h.get("corpus") != "v2" or h.get("slot") != sl or st != (sl * L_slot if sl is not None else None) \
                or not isinstance(nch, int) or not 0 < nch <= L_slot or st + nch > spec["corpus_chars"] \
                or h.get("books") != overlapping_books(spec, st, st + nch):
            bad.append(f"{tag}: haystack {({k: h.get(k) for k in ('corpus', 'slot', 'start', 'n_chars')})} (slot {sl})")
        else:
            slots[sl] += 1
        res = r.get("res")
        if not isinstance(res, dict) or set(res) != want_methods:
            bad.append(f"{tag}: methods {sorted(res) if isinstance(res, dict) else res}")
            continue
        for m, v in res.items():
            if any(f not in v for f in ("out", "score", "strict")):
                bad.append(f"{tag} {m}: missing out/score/strict")
                continue
            if r.get("answers") and float(v["strict"]) != float(score_strict(r["task"], v["out"], r["answers"])):
                bad.append(f"{tag} {m}: strict does not re-compute from out")
            if m.startswith("dense"):
                continue
            t = v.get("tokens")
            if not isinstance(t, dict) or any(f not in t for f in COUNTERS):
                bad.append(f"{tag} {m}: token counters missing")
                continue
            if not (isinstance(t["calls"], int) and t["calls"] > 0 and t["budget_checked_calls"] == t["calls"]):
                bad.append(f"{tag} {m}: calls {t['calls']} checked {t['budget_checked_calls']}")
            if t["budget_violations"] or t["max_abs_budget_error"] or t["fixed_region_violations"]:
                bad.append(f"{tag} {m}: budget/fixed-region violation")
    bad += [f"slot {s} used {c} times" for s, c in slots.items() if c > 1]
    return rows, missing, bad


def main(spec_path, path, partial):
    spec = json.load(open(spec_path))
    rows, missing, bad = check(spec, path, partial)
    print(f"rows {len(rows)}; missing {len(missing)}; problems {len(bad)}")
    for b in bad[:30]:
        print("  -", b)
    if bad:
        print("PHASE-2 GATE FAILED")
        sys.exit(2)
    print("PHASE-2 GATE PASSED" if not missing else f"PRESENT ROWS VALID — INCOMPLETE ({len(missing)} keys missing; "
          f"not an analysis licence)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--partial" in sys.argv[3:])
