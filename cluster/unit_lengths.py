"""Retrieval-unit lengths on frozen RULER data (plan v1.4 §3.4, reviewer T1) — mechanical, decided before any prediction.

Unit = the minimal key->value span a correct answer must read, measured in tokens of the exact prompt the predictor
feeds (input + answer_prefix, add_special_tokens=True as in src/ruler_pred.py):
  - niah_* : per answer value v, the needle "... for {key} is: {v}." -> span from the first character of {key} to the
             last character of {v};
  - vt     : per answer variable X, its assignment "VAR X = <rhs>" (rhs = the number or "VAR Y") -> span from X to the
             end of <rhs>; the last occurrence is used (the in-context example chain comes first);
  - cwe, fwe, qa_1, qa_2: no single unit (group "no_unit").
A unit's token length l = number of tokens whose character offsets overlap the span. Task group at block B (rule frozen
in plan coverage_rule): "coverable" iff every unit of every evaluated sample (frozen lines 0..n-1) has l <= B/2 + 1
(theory L1: some block of one of two grids offset by B/2 then contains the unit); otherwise "not_coverable".
Per unit, at the first decode call (n = prompt length L; sparse_attn._fixed_end/_layout with base_shift 0, local and
layout unit U from the plan protocol), with h = B/2 and fe = U + max(0, (L - local)//U - 1)*U:
  X0 = the unit straddles a grid-0 boundary (s//B != e//B), X1 = it straddles a grid-1 boundary ((s-h)//B != (e-h)//B);
  class "fixed"  : inside the always-read region [0, U) or [fe, L);
        "in0"    : not X0 and inside grid-0 candidates [U, fe)            (readable by the single grid);
        "repair" : X0, not X1 and inside grid-1 candidates [U+h, fe-h)     (the unit L1 says TWIN-GRID can recover);
        "out"    : anything else (straddles both grids, or lies in an edge strip) (outside the lemma's range).
One unit per answer reference, aligned with smp["outputs"] (so unit k is scored by whether reference k appears in the
prediction — the official string_match_all credit); a reference whose needle occurs more than once takes its most
readable occurrence (fixed < in0 < repair < out). pi_t (gain law, reviewer U1) = share of "repair" units. Multi-unit
tasks are scored per unit by the official metric, but resampling must be by sample (units of one sample are not
independent). v3 adds, per unit, vlen = value tokens and rv = value tokens after the first grid-0 boundary inside the
unit (None if not X0): rv = vlen means the cut falls before the value, 0 < rv < vlen that it cuts the value (P-asym). An extraction failure (no unit found where one
must exist) is an error, never a guess.
Usage: unit_lengths.py --data-dir DIR --tokenizer MODEL_DIR --n N --block B [--local 64] [--json OUT]"""
import argparse
import hashlib
import json
import os
import re
import sys

NO_UNIT = ("cwe", "fwe", "qa_1", "qa_2")
VERSION = "unit_lengths v3"


def unit_spans(task, smp):
    """Per answer reference (in the order of smp["outputs"]): the list of (a, b, v) character offsets of its unit
    occurrences (span [a, b), value starting at v); [] for no-unit tasks."""
    text, outs = smp["input"], smp["outputs"]
    if task in NO_UNIT:
        return []
    refs = []
    if task.startswith("niah"):
        for v in outs:
            ms = list(re.finditer(r"special magic [^\n:]*? for ([^\n:]+?) is: " + re.escape(v) + r"\.", text))
            if not ms:
                raise ValueError(f"{task}: needle for value {v!r} not found")
            refs.append([(m.start(1), m.end() - 1, m.end() - 1 - len(v)) for m in ms])
    elif task == "vt":
        for x in outs:
            ms = list(re.finditer(r"VAR (" + re.escape(x) + r") = (VAR [A-Za-z]+|\d+)", text))
            if not ms:
                raise ValueError(f"vt: assignment of {x!r} not found")
            refs.append([(ms[-1].start(1), ms[-1].end(), ms[-1].start(2))])
    else:
        raise ValueError(f"unknown task {task}")
    return refs


def token_spans(offsets, spans):
    """Inclusive token index range [s, e] overlapping each character span."""
    res = []
    for a, b in spans:
        idx = [i for i, (x, y) in enumerate(offsets) if y > a and x < b and y > x]
        if not idx:
            raise ValueError("unit span maps to no tokens")
        res.append((idx[0], idx[-1]))
    return res


RANK = {"fixed": 0, "in0": 1, "repair": 2, "out": 3}


def unit_class(s, e, block, fe, unit=0):
    h, U = block // 2, unit or block
    x0, x1 = s // block != e // block, (s - h) // block != (e - h) // block
    if e < U or s >= fe:
        return x0, x1, "fixed"
    if not x0 and s >= U and e < fe:
        return x0, x1, "in0"
    if x0 and not x1 and s >= U + h and e < fe - h:
        return x0, x1, "repair"
    return x0, x1, "out"


def measure(data_dir, tokenizer_dir, n, block, tasks, local=64, unit=0):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    if not tok.is_fast:
        raise SystemExit("a fast tokenizer (offset mapping) is required")
    U = unit or block
    out = dict(version=VERSION, block=block, unit=U, local=local, threshold=block // 2 + 1, n=n,
               tokenizer_json_sha256=hashlib.sha256(open(os.path.join(tokenizer_dir, "tokenizer.json"), "rb")
                                                    .read()).hexdigest(), tasks={})
    for task in tasks:
        f = os.path.join(data_dir, task, "validation.jsonl")
        lines = [json.loads(l) for l in open(f)][:n]
        if len(lines) < n:
            raise SystemExit(f"{task}: {len(lines)} < n={n} lines")
        samples = []
        for smp in lines:
            refs = unit_spans(task, smp)
            if not refs:
                samples.append(dict(units=0, lens=[], x0=[], x1=[], cls=[]))
                continue
            enc = tok(smp["input"] + smp.get("answer_prefix", ""), add_special_tokens=True,
                      return_offsets_mapping=True)
            L = len(enc["input_ids"])
            fe = U + max(0, (L - local) // U - 1) * U
            ts, cl, vs = [], [], []
            for occ in refs:                  # one unit per reference; repeated occurrence: the most readable one
                tsp = token_spans(enc["offset_mapping"], [(a, b) for a, b, _ in occ])
                vst = [token_spans(enc["offset_mapping"], [(v, b)])[0][0] for _, b, v in occ]
                cand = [(RANK[c[2]], (s, e), c, v0) for (s, e), v0 in zip(tsp, vst)
                        for c in [unit_class(s, e, block, fe, U)]]
                _, se, c, v0 = min(cand)
                ts.append(se)
                cl.append(c)
                vs.append(v0)
            cuts = [(s // block + 1) * block if c[0] else None for (s, e), c in zip(ts, cl)]
            samples.append(dict(units=len(ts), L=L, lens=[e - s + 1 for s, e in ts], x0=[c[0] for c in cl],
                                x1=[c[1] for c in cl], cls=[c[2] for c in cl],
                                vlen=[e - v0 + 1 for (s, e), v0 in zip(ts, vs)],
                                rv=[None if cut is None else e - max(v0, cut) + 1 for (s, e), v0, cut in zip(ts, vs, cuts)]))
        if task in NO_UNIT:
            group = "no_unit"
        else:
            group = "coverable" if max(max(x["lens"]) for x in samples) <= block // 2 + 1 else "not_coverable"
        out["tasks"][task] = dict(group=group, data_sha256=hashlib.sha256(open(f, "rb").read()).hexdigest(),
                                  samples=samples)
    return out


def outs_unique(smp):
    return list(dict.fromkeys(smp["outputs"]))


def summary(res):
    rows = []
    for t, r in res["tasks"].items():
        ls = [x for s in r["samples"] for x in s["lens"]]
        cl = [c for s in r["samples"] for c in s["cls"]]
        us = [s["units"] for s in r["samples"]]
        cnt = {k: cl.count(k) for k in ("in0", "repair", "out", "fixed")}
        civ = sum(1 for smp in r["samples"] for rv, vl in zip(smp.get("rv", []), smp.get("vlen", []))
                  if rv is not None and rv < vl)
        rows.append(f"  {t:16s} {r['group']:13s} units/sample {min(us)}-{max(us)}  "
                    + (f"l {min(ls)}-{max(ls)} (mean {sum(ls)/len(ls):.1f})  units in0/repair/out/fixed "
                       f"{cnt['in0']}/{cnt['repair']}/{cnt['out']}/{cnt['fixed']}  pi(repair) {cnt['repair']/len(cl):.3f}"
                       f"  cut inside value {civ}"
                       if ls else "-"))
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--local", type=int, default=None)
    ap.add_argument("--unit", type=int, default=None, help="layout unit (default: plan protocol --layout-unit)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    plan = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan_v1.json")))
    tasks = a.tasks.split(",") if a.tasks else plan["tasks"]
    try:
        local = a.local if a.local is not None else int(plan["protocol"]["argv_values"]["--local"])
        unit = a.unit if a.unit is not None else int(plan["protocol"]["argv_values"]["--layout-unit"])
        res = measure(a.data_dir, a.tokenizer, a.n, a.block, tasks, local, unit)
    except ValueError as e:
        print(f"REFUSED: {e}")
        sys.exit(3)
    print(f"{VERSION}: block {a.block}, coverable iff max unit length <= {a.block // 2 + 1}; n={a.n}; "
          f"tokenizer.json {res['tokenizer_json_sha256'][:16]}; local {res['local']}; layout unit {res['unit']}")
    print(summary(res))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
