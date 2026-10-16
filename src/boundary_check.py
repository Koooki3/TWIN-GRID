"""Does a wrong answer coincide with the answer value straddling a block boundary?

For each sample, find the token span of each target value; it 'straddles' if its first and
last tokens fall in different B-token blocks (block grid starting at position 0, as in
sparse_attn._layout). Report accuracy per method split by straddle / no straddle.
"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tasks import make_sample  # noqa: E402


def main(path, methods, B=32, model="/home/kooki3/models/qwen2.5-1.5b-instruct"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    acc = defaultdict(lambda: [0.0, 0])
    nstr = [0, 0]
    for line in open(path):
        r = json.loads(line)
        smp = make_sample(r["task"], r["ctx"], tok, r["seed"])
        enc = tok(smp["prompt"], return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        straddle = False
        for cs, ce, tgt in smp["spans"]:
            if not tgt:
                continue
            sent = smp["prompt"][cs:ce]
            for v in r["answers"]:
                j = sent.find(v)
                if j < 0:
                    continue
                a0, a1 = cs + j, cs + j + len(v)
                toks = [i for i, (s, e) in enumerate(offs) if s < a1 and e > a0]
                if toks and toks[0] // B != toks[-1] // B:
                    straddle = True
        nstr[straddle] += 1
        for m in methods:
            if m in r["res"]:
                k = (m, r["task"] == "multivalue", straddle)
                acc[k][0] += r["res"][m]["score"]
                acc[k][1] += 1
    print(f"samples: straddle={nstr[1]}  no straddle={nstr[0]}  (block size {B})")
    for m in methods:
        for mv in (False, True):
            a = acc[(m, mv, True)]
            b = acc[(m, mv, False)]
            name = "multivalue" if mv else "single/multikey"
            print(f"{m:14s} {name:16s} straddle {a[0] / max(a[1], 1):.2f} (n={a[1]})  "
                  f"no straddle {b[0] / max(b[1], 1):.2f} (n={b[1]})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","), B=int(sys.argv[3]) if len(sys.argv) > 3 else 32)
