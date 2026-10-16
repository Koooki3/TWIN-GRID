"""Delimiter-chunking development analysis (exploratory, descriptive; docs/THEORY_ADDENDUM_ZH.md §5; reviewer batch 5).
Written and committed BEFORE the delim run starts. Arms on the same 7 unit tasks x 20 frozen lines (Qwen2.5-1.5B, 16k,
budget 512, layout unit 32): cobs and cobs+g2f from the P-grid run (runs/ruler/pgrid_pred/b32), and cobs (fidelity
re-run) + delim_cobs from src/delim_dev.py.
Per unit (one per answer reference; correct iff the reference occurs in the post-processed prediction):
  - 32-grid class at the first decode call (cluster/unit_lengths.py v3, U = 32): in0 / repair / out / fixed;
  - TWIN-GRID coverage: class in {fixed, in0, repair} (whole in one candidate block of some grid, or always read);
  - delim coverage: the unit's token span lies inside one delim chunk (delim_attn.chunks at n = L) or in the fixed region.
Reports: (0) runner fidelity — cobs re-run pred == P-grid cobs pred, count; token_sha of the re-run equals the
P-grid row's and every sidecar token_sha is a delim row's (else exit 2; reviewer batch 6, Z3); (1) unit accuracy per arm, overall and per
task; (2) by 32-grid class; (3) by delim coverage; (4) coverage rates per task (TWIN-GRID vs delim); (5) delim fill share
(mean / max over samples) and chunks per candidate token vs 1/32 (single grid) and 2/32 (TWIN-GRID); differences
delim - (cobs+g2f) with sample-level bootstrap within task (2000). No decision rule.
Usage: delim_dev_analysis.py DATA_DIR PGRID_B32_DIR DELIM_DIR TOKENIZER [N]"""
import json
import os
import sys

import numpy as np
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "cluster"))
import delim_attn  # noqa: E402
from aggregate import postprocess_pred  # noqa: E402
from unit_lengths import measure, token_spans, unit_spans  # noqa: E402

TASKS = ("niah_single_1", "niah_single_2", "niah_multikey_1", "niah_multikey_2", "niah_multivalue", "niah_multiquery",
         "vt")
ARMS = ("cobs", "cobs+g2f", "delim_cobs")


def load(path):
    return {json.loads(l)["sample_uid"]: json.loads(l) for l in open(path)} if os.path.exists(path) else {}


def main(data, pg, dl, tokdir, n):
    tok = AutoTokenizer.from_pretrained(tokdir)
    u32 = measure(data, tokdir, n, 32, TASKS, 64, 32)
    recs, fid, stats, tokbad = [], [0, 0], [], []
    for t in TASKS:
        lines = [json.loads(l) for l in open(os.path.join(data, t, "validation.jsonl"))][:n]
        rows = {"cobs": load(f"{pg}/cobs@512/{t}.jsonl"), "cobs+g2f": load(f"{pg}/cobs+g2f@512/{t}.jsonl"),
                "delim_cobs": load(f"{dl}/delim_cobs@512/{t}.jsonl"), "_cobs_rerun": load(f"{dl}/cobs@512/{t}.jsonl")}
        stats += [json.loads(l) for l in open(f"{dl}/delim_stats.{t}.jsonl")] if os.path.exists(
            f"{dl}/delim_stats.{t}.jsonl") else []
        for uid, r in rows["_cobs_rerun"].items():
            if uid in rows["cobs"]:
                fid[0] += r["pred"] == rows["cobs"][uid]["pred"]
                fid[1] += 1
                if r.get("token_sha") != rows["cobs"][uid].get("token_sha"):       # reviewer batch 6, Z3
                    tokbad.append(f"{uid}: token_sha re-run != P-grid")
        dsha = {r.get("token_sha") for r in rows["delim_cobs"].values()}
        for st in ([json.loads(l) for l in open(f"{dl}/delim_stats.{t}.jsonl")]
                   if os.path.exists(f"{dl}/delim_stats.{t}.jsonl") else []):
            if st["token_sha"] not in dsha:
                tokbad.append(f"{t}: sidecar token_sha {st['token_sha']} not among delim rows")
        for i, ln in enumerate(lines):
            uid = next((u for u in rows["delim_cobs"] if u.endswith(f":{t}:{i}")), None)
            if uid is None or any(uid not in rows[a] for a in ARMS):
                continue
            ids = tok(ln["input"] + ln.get("answer_prefix", ""), add_special_tokens=True,
                      return_offsets_mapping=True)
            L = len(ids["input_ids"])
            fe = 32 + max(0, (L - 64) // 32 - 1) * 32
            delim_attn.DELIM.ends = delim_attn.sentence_ends(tok, ids["input_ids"])
            delim_attn.DELIM.cache = {}
            cid, _, _ = delim_attn.chunks(32, fe, 32)
            refs = unit_spans(t, ln)
            smp = u32["tasks"][t]["samples"][i]
            for k, ref in enumerate(ln["outputs"]):
                spans = token_spans(ids["offset_mapping"], [(a, b) for a, b, _ in refs[k]])
                cov_d = any(e < 32 or s >= fe or (s >= 32 and e < fe and int(cid[s - 32]) == int(cid[e - 32]))
                            for s, e in spans)
                ok = {a: int(ref.lower() in postprocess_pred(rows[a][uid]["pred"]).lower()) for a in ARMS}
                recs.append(dict(task=t, i=i, cls=smp["cls"][k], cov_t=smp["cls"][k] != "out", cov_d=cov_d, ok=ok))
    print(f"delim development (exploratory): {len(recs)} units, {len({(r['task'], r['i']) for r in recs})} samples")
    print(f"(0) runner fidelity: cobs re-run pred == P-grid cobs pred in {fid[0]}/{fid[1]} samples; token_sha "
          f"checks (re-run vs P-grid, sidecar vs delim rows): {'ok' if not tokbad else 'FAILED'}")
    if tokbad:
        print("TOKEN_SHA CHECK FAILED:", *tokbad[:20], sep="\n  ")
        sys.exit(2)
    rng = np.random.default_rng(0)
    samples = sorted({(r["task"], r["i"]) for r in recs})
    by = {s: [r for r in recs if (r["task"], r["i"]) == s] for s in samples}

    def ci(filt):
        ds = []
        for _ in range(2000):
            pick = []
            for t in TASKS:
                ss = [s for s in samples if s[0] == t]
                pick += [ss[j] for j in rng.integers(0, len(ss), len(ss))] if ss else []
            u = [r for s in pick for r in by[s] if filt(r)]
            if u:
                ds.append(np.mean([r["ok"]["delim_cobs"] - r["ok"]["cobs+g2f"] for r in u]))
        return np.percentile(ds, [2.5, 97.5]) if ds else (np.nan, np.nan)

    def line(name, filt):
        u = [r for r in recs if filt(r)]
        if not u:
            return
        acc = {a: np.mean([r["ok"][a] for r in u]) for a in ARMS}
        lo, hi = ci(filt)
        print(f"  {name:28s} units {len(u):4d}: " + "  ".join(f"{a} {acc[a]:.3f}" for a in ARMS)
              + f"  | delim - g2f {acc['delim_cobs'] - acc['cobs+g2f']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("(1) overall and per task")
    line("all", lambda r: True)
    for t in TASKS:
        line(t, lambda r, t=t: r["task"] == t)
    print("(2) by 32-grid class")
    for c in ("in0", "repair", "out", "fixed"):
        line(f"class {c}", lambda r, c=c: r["cls"] == c)
    print("(3) by delim coverage")
    line("delim-covered", lambda r: r["cov_d"])
    line("not delim-covered", lambda r: not r["cov_d"])
    print("(4) coverage rates per task: TWIN-GRID (class != out) vs delim (unit inside one chunk or fixed)")
    for t in TASKS:
        u = [r for r in recs if r["task"] == t]
        if u:
            print(f"  {t:16s} units {len(u):4d}: TWIN-GRID {np.mean([r['cov_t'] for r in u]):.3f}  "
                  f"delim {np.mean([r['cov_d'] for r in u]):.3f}")
    if stats:
        fm = [s["fill_share_mean"] for s in stats]
        print(f"(5) delim fill share: mean over samples {np.mean(fm):.4f}, max per call {max(s['fill_share_max'] for s in stats):.4f}; "
              f"chunks per candidate token {np.mean([s['chunks_per_cand_token'] for s in stats]):.4f} "
              f"(single grid {1 / 32:.4f}, TWIN-GRID {2 / 32:.4f}); samples {len(stats)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 20)
