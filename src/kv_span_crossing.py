"""Descriptive (development data): does the KEY→VALUE span ("for {key} is: {value}", from the first key token to the
last value token) cross a grid-0 / grid-1 block boundary, and how does accuracy split by that? Rebuilds each prompt with
the frozen generator (tokenizer only, no model) and checks the prompt hash against the run row.
Usage: kv_span_crossing.py RUN.jsonl [ARM ...]"""
import hashlib
import json
import sys

from transformers import AutoTokenizer

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tasks import make_sample  # noqa: E402

B = 32


def spans(tok, smp):
    enc = tok(smp["prompt"], return_offsets_mapping=True)
    offs = enc.offset_mapping
    for cs, ce, tgt in smp["spans"]:
        if not tgt:
            continue
        sent = smp["prompt"][cs:ce]
        k0 = cs + sent.find(" for ") + len(" for ")
        v0 = cs + sent.find(smp["answers"][0])
        v1 = v0 + len(smp["answers"][0])
        kt = next(i for i, (a, b) in enumerate(offs) if a <= k0 < b)
        vt = [i for i, (a, b) in enumerate(offs) if a < v1 and b > v0]
        return kt, vt[0], vt[-1]


def cross(a, b, off):
    return (a - off) // B != (b - off) // B


def main(path, arms):
    tok = AutoTokenizer.from_pretrained("/home/kooki3/models/qwen2.5-1.5b-instruct")
    rows = [json.loads(l) for l in open(path)]
    arms = arms or sorted({m.split("@")[0] for r in rows for m in r["res"]})
    tab = []
    for r in rows:
        smp = make_sample(r["task"], r["ctx"], tok, r["seed"])
        if hashlib.sha256(smp["prompt"].encode()).hexdigest()[:16] != r["prompt_sha"]:
            sys.exit(f"prompt hash mismatch {r['task']} {r['seed']}")
        kt, v_first, v_last = spans(tok, smp)
        tab.append(dict(r=r, kv_len=v_last - kt + 1, v_g0=cross(v_first, v_last, 0),
                        kv_g0=cross(kt, v_last, 0), kv_g1=cross(kt, v_last, B // 2)))
    n = len(tab)
    print(f"prompts {n}; key→value span length (tokens): min {min(t['kv_len'] for t in tab)} "
          f"max {max(t['kv_len'] for t in tab)}; value crosses g0 {sum(t['v_g0'] for t in tab)}; "
          f"key→value crosses g0 {sum(t['kv_g0'] for t in tab)}; crosses both grids {sum(t['kv_g0'] and t['kv_g1'] for t in tab)}")
    groups = {"kv inside g0": [t for t in tab if not t["kv_g0"]],
              "kv crosses g0, inside g1": [t for t in tab if t["kv_g0"] and not t["kv_g1"]],
              "kv crosses both": [t for t in tab if t["kv_g0"] and t["kv_g1"]]}
    for arm in arms:
        key = f"{arm}@512"
        if key not in tab[0]["r"]["res"]:
            continue
        print(f"  {arm:14s} " + "  ".join(f"{g}: {sum(int(t['r']['res'][key]['strict']) for t in ts)}/{len(ts)}"
                                          for g, ts in groups.items()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
