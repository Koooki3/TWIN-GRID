"""Registered descriptive prediction P-kv (theory v8 §7), written 2026-10-16 while the prereg-v3 confirmation run was in
progress (about 70/300 rows written, results NOT looked at). It is NOT part of the frozen prereg-v3 analysis and does
not change it; it is a separately registered secondary, descriptive analysis on the same confirmation file.

Origin (post hoc, development data runs/phase1_g2f_dev.jsonl, 80 prompts): the KEY→VALUE span of the target needle
(first token of "{key}" in "... for {key} is: {value}." through the last value token; 11–12 tokens) crosses a grid-0
boundary in 25/80 prompts (≈ (ℓ−1)/B); COBS-style strict accuracy 48/55 inside vs 16/25 crossing; with +g2f 48/55 vs
24/25 — the whole +g2f gain sat on kv-crossing prompts (nf_pt1: +5 crossing, −1 inside; Quest: +1 crossing, +5 inside).

Predictions on the confirmation file (300 prompts, fresh seeds, v2 haystack), fixed now:
  P1 the share of prompts whose KV span crosses grid 0 lies in [0.22, 0.42] (theory (ℓ−1)/B with ℓ ≈ 11–12, B = 32);
  P2 COBS-style strict accuracy is lower on kv-crossing than on kv-inside prompts (difference of proportions with
     a 95% Newcombe hybrid-score interval (method 10 unpaired = Newcombe 1998 Stat Med 17:873 method 10) excluding 0);
  P3 the +g2f gain for COBS-style is concentrated on kv-crossing prompts: interaction
     I = [acc(cobs+g2f) − acc(cobs)]_cross − [acc(cobs+g2f) − acc(cobs)]_inside > 0, with a 95% prompt-bootstrap
     interval (stratified by crossing status, 10,000 resamples, seed 0) excluding 0;
  P4 (exploratory, no decision) the same interaction for nf_pt1 and Quest is reported.
Each of P1–P3 is reported as "supported" / "not supported" by these rules only; no other cut-offs are used later.
The prompt is rebuilt with the frozen generator from the row's (task, seed, hay.slot) and its hash must equal the row's
prompt_sha (exit 2 otherwise). Tokenizer only; no model.
Usage: kv_prediction.py PHASE2.jsonl"""
import hashlib
import json
import math
import random
import sys

from transformers import AutoTokenizer

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tasks import make_sample  # noqa: E402

B = 32
Z = 1.959963984540054


def kv_span(tok, smp):
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
            return kt, vt[-1]


def wilson(x, n):
    p, d = x / n, 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def newcombe_unpaired(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def main(path):
    tok = AutoTokenizer.from_pretrained("/home/kooki3/models/qwen2.5-1.5b-instruct")
    rows = [json.loads(l) for l in open(path)]
    tab = []
    for r in rows:
        slot = (r.get("hay") or {}).get("slot")
        smp = make_sample(r["task"], r["ctx"], tok, r["seed"], slot)
        if hashlib.sha256(smp["prompt"].encode()).hexdigest()[:16] != r["prompt_sha"]:
            print(f"prompt hash mismatch {r['task']} {r['seed']}")
            sys.exit(2)
        kt, vl = kv_span(tok, smp)
        tab.append(dict(cross=kt // B != vl // B, s={m.split("@")[0]: int(v["strict"]) for m, v in r["res"].items()}))
    n = len(tab)
    cr = [t for t in tab if t["cross"]]
    ins = [t for t in tab if not t["cross"]]
    share = len(cr) / n
    print(f"prompts {n}; kv-crossing {len(cr)} ({share:.3f}); inside {len(ins)}")
    print(f"P1 share in [0.22, 0.42]: {'supported' if 0.22 <= share <= 0.42 else 'not supported'}")
    xi, xc = sum(t["s"]["cobs"] for t in ins), sum(t["s"]["cobs"] for t in cr)
    d, lo, hi = newcombe_unpaired(xi, len(ins), xc, len(cr))
    print(f"P2 cobs inside {xi}/{len(ins)} vs crossing {xc}/{len(cr)}: diff {d:+.3f} [{lo:+.3f}, {hi:+.3f}] -> "
          f"{'supported' if lo > 0 else 'not supported'}")
    rng = random.Random(0)

    def inter(c, i, base):
        g = lambda ts: sum(t["s"][base + "+g2f"] - t["s"][base] for t in ts) / len(ts)  # noqa: E731
        return g(c) - g(i)

    for base in ("cobs", "nf_pt1", "quest"):
        I = inter(cr, ins, base)
        bs = sorted(inter([rng.choice(cr) for _ in cr], [rng.choice(ins) for _ in ins], base) for _ in range(10000))
        lo, hi = bs[249], bs[9749]
        gc = sum(t["s"][base + "+g2f"] - t["s"][base] for t in cr)
        gi = sum(t["s"][base + "+g2f"] - t["s"][base] for t in ins)
        tag = "P3" if base == "cobs" else "P4"
        verdict = ("supported" if lo > 0 else "not supported") if base == "cobs" else "exploratory"
        print(f"{tag} {base}: net gain crossing {gc:+d}/{len(cr)}, inside {gi:+d}/{len(ins)}; interaction {I:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}] -> {verdict}")


if __name__ == "__main__":
    main(sys.argv[1])
