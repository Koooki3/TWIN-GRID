"""Post hoc, hypothesis-generating diagnostics on the frozen confirmation file (reviewer batch 1, T4 and T5). Written
AFTER the confirmation results were seen; nothing here is confirmatory, and it does not change prereg v3 or P-kv.
Tokenizer only; prompts rebuilt with the frozen generator and checked against prompt_sha (exit 2 otherwise).

T4 (residual): every prompt's KEY->VALUE unit (as in src/kv_prediction.py) is classed exactly as
cluster/unit_lengths.py v2 does at the first decode call (n = L, local and B from the row): in0 / repair (straddles
grid 0, inside one grid-1 candidate block) / out / fixed. Question: are the straddling prompts that +g2f leaves
wrong the "out" ones (outside the covering lemma's range)? Table: class x (baseline, +g2f) strict outcomes per scorer.
T5 (cut offset): on grid-0 straddling units, the cut splits the unit into a left part (holding the key phrase) and a
right part; the right part holds r_v value tokens. Two-level fraction sensitivity (F) depends on the fragment sizes
only; the read requirement (R) predicts failure to grow with r_v at matched fragment sizes. Table: baseline failure by
(left size, r_v). Small counts; association only.
Usage: kv_residual.py FILE.jsonl [FILE.jsonl ...]"""
import hashlib
import json
import sys
from collections import Counter, defaultdict

from transformers import AutoTokenizer

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tasks import make_sample  # noqa: E402
sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/cluster")
from unit_lengths import unit_class  # noqa: E402

SCORERS = ("cobs", "nf_pt1", "quest")


def spans(tok, smp):
    enc = tok(smp["prompt"], return_offsets_mapping=True)
    offs = enc.offset_mapping
    for cs, ce, tgt in smp["spans"]:
        if tgt:
            sent = smp["prompt"][cs:ce]
            k0 = cs + sent.find(" for ") + len(" for ")
            v0 = cs + sent.find(smp["answers"][0])
            v1 = v0 + len(smp["answers"][0])
            kt = next(i for i, (a, b) in enumerate(offs) if a <= k0 < b)
            vt = [i for i, (a, b) in enumerate(offs) if a < v1 and b > v0]
            return kt, vt[0], vt[-1], len(enc.input_ids)


def main(paths):
    tok = AutoTokenizer.from_pretrained("/home/kooki3/models/qwen2.5-1.5b-instruct")
    t4 = defaultdict(Counter)
    t5 = defaultdict(Counter)
    mir = defaultdict(Counter)
    lens = Counter()
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            smp = make_sample(r["task"], r["ctx"], tok, r["seed"], (r.get("hay") or {}).get("slot"))
            if hashlib.sha256(smp["prompt"].encode()).hexdigest()[:16] != r["prompt_sha"]:
                print(f"prompt hash mismatch {r['task']} {r['seed']}")
                sys.exit(2)
            kt, v0, vl, L = spans(tok, smp)
            if L != r["L"]:
                print(f"token count {L} != row L {r['L']} ({r['task']} {r['seed']})")
                sys.exit(2)
            B, local = r["block"], r["local"]
            fe = B + max(0, (L - local) // B - 1) * B
            x0, x1, cl = unit_class(kt, vl, B, fe)
            s = {m.split("@")[0]: int(v["strict"]) for m, v in r["res"].items()}
            lens[vl - kt + 1] += 1
            for sc in SCORERS:
                t4[path, cl][sc, s[sc], s[sc + "+g2f"]] += 1
            if x0:
                cut = (kt // B + 1) * B                      # first grid-0 boundary inside the unit
                left, rv = cut - kt, sum(1 for i in range(v0, vl + 1) if i >= cut)
                right = vl - cut + 1
                side = "right" if right < left else ("left" if left < right else "equal")
                for sc in SCORERS:
                    t5[sc][left, rv, s[sc]] += 1
                    mir[sc][min(left, right), side, s[sc]] += 1
    print(f"unit length distribution {dict(sorted(lens.items()))}")
    print("T4: class x outcome (b = baseline strict, t = +g2f strict); per file")
    for (path, cl), c in sorted(t4.items()):
        n = sum(v for k, v in c.items() if k[0] == "cobs")
        row = []
        for sc in SCORERS:
            b = sum(v for k, v in c.items() if k[0] == sc and k[1])
            t = sum(v for k, v in c.items() if k[0] == sc and k[2])
            gain = sum(v for k, v in c.items() if k[0] == sc and k[2] and not k[1])
            loss = sum(v for k, v in c.items() if k[0] == sc and k[1] and not k[2])
            row.append(f"{sc} {b}->{t} (+{gain}/-{loss}) wrong after g2f {n - t}")
        print(f"  {path.rsplit('/', 1)[-1]:28s} {cl:7s} n {n:3d}: " + "; ".join(row))
    print("T5: grid-0 straddling units, baseline failures by (left part size incl. key phrase, value tokens right of"
          " the cut); all files pooled")
    for sc in SCORERS:
        cells = sorted({(k[0], k[1]) for k in t5[sc]})
        print(f"  {sc}: " + "; ".join(f"left {l} rv {v}: fail {t5[sc][l, v, 0]}/{t5[sc][l, v, 0] + t5[sc][l, v, 1]}"
                                     for l, v in cells))
        byv = defaultdict(lambda: [0, 0])
        for (l, v, ok), c in t5[sc].items():
            byv[v][0] += c * (1 - ok)
            byv[v][1] += c
        print(f"    by rv only: " + "; ".join(f"rv {v}: {f}/{n}" for v, (f, n) in sorted(byv.items())))
    print("T5 mirror: smaller fragment of size m on the LEFT (cut near the key start) vs on the RIGHT (value tail cut"
          " off); the two-level F model is symmetric in the fragment sizes")
    for sc in SCORERS:
        parts = []
        for m in range(1, 6):
            l_f, l_n = mir[sc][m, "left", 0], mir[sc][m, "left", 0] + mir[sc][m, "left", 1]
            r_f, r_n = mir[sc][m, "right", 0], mir[sc][m, "right", 0] + mir[sc][m, "right", 1]
            parts.append(f"m {m}: left {l_f}/{l_n} right {r_f}/{r_n}")
        tl = sum(mir[sc][m, "left", 0] for m in range(1, 4)), sum(mir[sc][m, "left", o] for m in range(1, 4) for o in (0, 1))
        tr = sum(mir[sc][m, "right", 0] for m in range(1, 4)), sum(mir[sc][m, "right", o] for m in range(1, 4) for o in (0, 1))
        print(f"  {sc}: " + "; ".join(parts) + f"  | m<=3: left {tl[0]}/{tl[1]} vs right {tr[0]}/{tr[1]}")


if __name__ == "__main__":
    main(sys.argv[1:])
