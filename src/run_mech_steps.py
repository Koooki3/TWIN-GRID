"""Per-step mechanism metrics along the DENSE greedy answer trajectory.

For each decode step of the dense answer, every scorer's selection is evaluated against the
oracle top-k (retrieval-weighted captured mass, needle-block hit). Tests whether failures of
calibrated scorers happen after the first answer token.
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import sparse_attn  # noqa: F401
from sparse_attn import S
from tasks import make_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/kooki3/models/qwen2.5-1.5b-instruct")
    ap.add_argument("--tasks", default="multikey,single_essay,multivalue")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--method", default="dense", help="decode with this scorer (on-policy trajectory)")
    ap.add_argument("--mask", default="sentence", choices=["sentence", "value"],
                    help="target tokens: whole needle sentence, or only the answer value digits")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S.local, S.rank, S.budget = 64, a.rank, a.budget
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="blocksparse").cuda().eval()
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    agg = {}
    for task in a.tasks.split(","):
        for i in range(a.n):
            smp = make_sample(task, a.ctx, tok, 1000 + i)
            enc = tok(smp["prompt"], return_offsets_mapping=True, return_tensors="pt")
            ids = enc.input_ids.cuda()
            L = ids.shape[1]
            offs = enc.offset_mapping[0]
            nmask = torch.zeros(L + 128, dtype=torch.bool)
            for (cs, ce, tgt) in smp["spans"]:
                if tgt:
                    if a.mask == "value":               # only the answer digits inside the needle
                        sent = smp["prompt"][cs:ce]
                        for v in smp["answers"]:
                            j = sent.find(v)
                            if j >= 0:
                                cs_v, ce_v = cs + j, cs + j + len(v)
                                nmask[:L] |= (offs[:, 0] < ce_v) & (offs[:, 1] > cs_v)
                    else:
                        nmask[:L] |= (offs[:, 0] < ce) & (offs[:, 1] > cs)
            S.needle_mask = nmask
            with torch.no_grad():
                S.method, S.summ, S.W, S.C = "dense", {}, {}, {}
                cache = DynamicCache()
                for st in range(0, L - 1, 4096):
                    model(ids[:, st:min(st + 4096, L - 1)], past_key_values=cache, use_cache=True, logits_to_keep=1)
                cur = ids[:, -1:]
                S.method = a.method                     # prefill was dense; decode on-policy
                S.mech = {}
                for step in range(a.max_steps):
                    S.mech_step = step
                    lg = model(cur, past_key_values=cache, use_cache=True).logits[:, -1]
                    nxt = lg.argmax(-1, keepdim=True)
                    if nxt.item() in eos:
                        break
                    cur = nxt
                for step, d in S.mech.items():
                    for m, r in d.items():
                        g = agg.setdefault(task, {}).setdefault(str(step), {}).setdefault(m, [0.0] * 5)
                        for j in range(5):
                            g[j] += r[j]
                S.mech = None
            del cache
            torch.cuda.empty_cache()
            print(task, i, flush=True)
    out = {t: {st: {m: dict(capture_w=v[0] / max(v[2], 1e-9), hit_w=v[1] / max(v[2], 1e-9),
                            capture_all_heads=v[3] / max(v[4], 1))
                    for m, v in d.items()} for st, d in steps.items()} for t, steps in agg.items()}
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
