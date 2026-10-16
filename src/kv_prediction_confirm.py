"""Confirmation entry for the registered prediction P-kv (Codex K1/K3). The ORIGINAL registered script
src/kv_prediction.py (sha256 0b80dc34d1b03e24b013ecefb7cd09a0df2b6ab22d57874c412e75c3aef1994a, commit 4122939) is not
modified; this wrapper adds input validation before it runs and extra descriptive tables after it.

Before any P-kv output (exit 2 on failure):
  - the frozen prereg-v3 gate: phase2_gate.check(SPEC, RUN, partial=False) with no problems and no missing keys
    (300 keys, methods/budget, v2 slots, identities, strict re-computation, frozen analysis files);
  - the confirmation job's ledger receipt (JOB_RECEIPT) belongs to THIS confirmation (Codex K1-a): id equals
    CONFIRM_JOB, project dual-channel-attention, label phase2-confirm, phase finished, status succeeded, exit_code 0;
    its command runs src/run_e2e.py with exactly SPEC.argv, whose --out is the analysed RUN path; its recorded
    input fingerprint of theory/phase2_spec.json equals the SPEC file's sha256. A re-run would be a new, separately
    registered attempt/id — never substituted silently;
  - an acceptance record (results/phase2_kv_acceptance.json) seals the RUN, SPEC and receipt sha256 and the hashes of
    the registered script, this entry and the frozen gate before any P-kv output;
  - for every row, the prompt rebuilt from (task, seed, hay.slot) reproduces BOTH prompt_sha and token_sha; the token
    ids hashed are those of the SAME offset-mapping encoding that defines the key→value span (Codex K1-b), and the
    span is computed from those offsets here (the registered script's own second encoding is then compared to the
    same ids);
  - the tokenizer is the SPEC model's (tokenizer.json short hash equals SPEC.code["model:tokenizer.json"]) and the
    block size used for crossing (32) equals SPEC.fixed.block.
Then — unless the pooled kv-crossing or kv-inside group is empty, in which case P2–P4 are printed as "insufficient
information / not estimable" and the registered script is not called (only P1 is reported) — the original
kv_prediction.main runs unchanged (P1–P4 labels mean "met / not met the registered descriptive rule"; they are not
multiplicity-controlled confirmations and "not met" is not evidence of no effect; a banner says so), followed by
the K3 tables: task × crossing denominators, base/plus success counts, paired gains and losses for cobs / nf_pt1 /
quest; empty strata are printed as "not estimable". The pooled P3 estimand stays the registered one; no task-
standardised interaction is computed here (it would need its own registration).
--dev runs everything except the gate and the job check on a development file and prints "DEV — NOT CONFIRMATORY".
Usage: kv_prediction_confirm.py SPEC.json RUN.jsonl JOB_RECEIPT.json | kv_prediction_confirm.py --dev RUN.jsonl"""
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kv_prediction  # noqa: E402  (the registered, unmodified script)
from tasks import make_sample  # noqa: E402

TOKENIZER = "/home/kooki3/models/qwen2.5-1.5b-instruct"
B = 32
CONFIRM_JOB = "ef183d150ca1415d85f3f31919a19e90"
ROOT = os.path.dirname(HERE)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def receipt_problems(rec, spec, spec_path, run_path):
    bad = []
    want = dict(id=CONFIRM_JOB, project="dual-channel-attention", label="phase2-confirm", phase="finished",
                status="succeeded", exit_code=0)
    bad += [f"receipt {k}={rec.get(k)!r} (want {v!r})" for k, v in want.items() if rec.get(k) != v]
    cmd = rec.get("command") or []
    runner = os.path.join(HERE, "run_e2e.py")
    if runner not in cmd or cmd[cmd.index(runner) + 1:] != spec["argv"]:
        bad.append("receipt command is not run_e2e.py with exactly SPEC.argv")
    argv = spec["argv"]
    if "--out" not in argv or os.path.abspath(argv[argv.index("--out") + 1]) != os.path.abspath(run_path):
        bad.append("SPEC --out is not the analysed RUN path")
    fps = {i.get("path"): (i.get("fingerprint") or {}).get("sha256") for i in rec.get("inputs") or []}
    if fps.get(os.path.abspath(spec_path)) != sha(spec_path):
        bad.append("receipt input fingerprint of the SPEC differs from the SPEC file")
    return bad


def span_from(smp, enc):
    """Key→value span from ONE encoding's offsets (same rule as the registered kv_span)."""
    offs = enc.offset_mapping
    for cs, ce, tgt in smp["spans"]:
        if tgt:
            sent = smp["prompt"][cs:ce]
            k0 = cs + sent.find(" for ") + len(" for ")
            v0 = cs + sent.find(smp["answers"][0])
            v1 = v0 + len(smp["answers"][0])
            kt = next(i for i, (a, b) in enumerate(offs) if a <= k0 < b)
            vt = [i for i, (a, b) in enumerate(offs) if a < v1 and b > v0]
            return kt, vt[-1]


def token_check(rows, tok):
    bad, tab = [], []
    for r in rows:
        smp = make_sample(r["task"], r["ctx"], tok, r["seed"], (r.get("hay") or {}).get("slot"))
        enc = tok(smp["prompt"], return_offsets_mapping=True)          # the one encoding used for ids AND span
        ids = np.array([enc.input_ids], dtype=np.int64)
        if hashlib.sha256(smp["prompt"].encode()).hexdigest()[:16] != r["prompt_sha"] or \
                hashlib.sha256(ids.tobytes()).hexdigest()[:16] != r["token_sha"]:
            bad.append(f"{r['task']}:{r['seed']} prompt/token identity")
            continue
        kt, vl = span_from(smp, enc)
        if kv_prediction.kv_span(tok, smp) != (kt, vl):                # the registered script's own encoding agrees
            bad.append(f"{r['task']}:{r['seed']} registered span differs from the verified encoding")
            continue
        tab.append(dict(task=r["task"], cross=kt // B != vl // B,
                        s={m.split("@")[0]: int(v["strict"]) for m, v in r["res"].items()}))
    return bad, tab


def k3(tab):
    print("\nK3 task × kv-crossing (descriptive; strict successes; paired gain = plus right & base wrong, loss = reverse)")
    cells = defaultdict(list)
    for t in tab:
        cells[(t["task"], t["cross"])].append(t)
    for task in sorted({t["task"] for t in tab}):
        for cr in (False, True):
            ts = cells[(task, cr)]
            name = f"{task:12s} {'crossing' if cr else 'inside  '}"
            if not ts:
                print(f"  {name}: n 0 — not estimable")
                continue
            parts = []
            for b in ("cobs", "nf_pt1", "quest"):
                base, plus = sum(t["s"][b] for t in ts), sum(t["s"][b + "+g2f"] for t in ts)
                gain = sum(t["s"][b + "+g2f"] and not t["s"][b] for t in ts)
                loss = sum(t["s"][b] and not t["s"][b + "+g2f"] for t in ts)
                parts.append(f"{b} {base}->{plus} (+{gain}/-{loss})")
            print(f"  {name}: n {len(ts)}; " + "; ".join(parts))


def main(argv):
    dev = argv[0] == "--dev"
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    if dev:
        path = argv[1]
        print("DEV — NOT CONFIRMATORY")
    else:
        spec_path, path, receipt_path = argv[:3]
        spec = json.load(open(spec_path))
        import phase2_gate
        _, missing, bad = phase2_gate.check(spec, path, partial=False)
        rec = json.load(open(receipt_path))
        bad += receipt_problems(rec, spec, spec_path, path)
        if hashlib.sha256(open(f"{TOKENIZER}/tokenizer.json", "rb").read()).hexdigest()[:16] != \
                spec["code"].get("model:tokenizer.json"):
            bad.append("tokenizer differs from the SPEC model")
        if spec["fixed"]["block"] != B:
            bad.append("block size differs from the SPEC")
        if bad or missing:
            print("REFUSED before any P-kv output:", *bad[:10], f"missing keys {len(missing)}", sep="\n  ")
            sys.exit(2)
    rows = [json.loads(l) for l in open(path)]
    bad, tab = token_check(rows, tok)
    if bad:
        print("REFUSED: token identity", *bad[:10], sep="\n  ")
        sys.exit(2)
    if not dev:
        acc = dict(run=os.path.abspath(path), run_sha256=sha(path), spec_sha256=sha(spec_path),
                   receipt=os.path.abspath(receipt_path), receipt_sha256=sha(receipt_path), job=CONFIRM_JOB,
                   kv_prediction_sha256=sha(os.path.join(HERE, "kv_prediction.py")),
                   kv_prediction_confirm_sha256=sha(os.path.abspath(__file__)),
                   phase2_gate_sha256=sha(os.path.join(HERE, "phase2_gate.py")))
        json.dump(acc, open(os.path.join(ROOT, "results", "phase2_kv_acceptance.json"), "w"), indent=1)
        print("acceptance record written: results/phase2_kv_acceptance.json")
    print("NOTE: P-kv labels only say whether a REGISTERED DESCRIPTIVE RULE was met; intervals are not adjusted for "
          "multiplicity; 'not supported' is not evidence of no effect.")
    n_cross = sum(t["cross"] for t in tab)
    if n_cross == 0 or n_cross == len(tab):
        print(f"prompts {len(tab)}; kv-crossing {n_cross}. P1 share {n_cross / len(tab):.3f} "
              f"({'supported' if 0.22 <= n_cross / len(tab) <= 0.42 else 'not supported'}). "
              "P2–P4: insufficient information / not estimable (an empty pooled group); registered script not called.")
    else:
        print("== registered P-kv (original script, unchanged) ==", flush=True)
        kv_prediction.main(path)
    k3(tab)
    if dev:
        print("\nDEV — NOT CONFIRMATORY")


if __name__ == "__main__":
    main(sys.argv[1:])
