"""H1 and outlier-hit diagnostics: one dense prefill + one decode step per sample and budget."""
import argparse
import json
import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import sparse_attn  # noqa: F401  (registers the attention backend)
from sparse_attn import S
from tasks import make_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/kooki3/models/qwen2.5-1.5b-instruct")
    ap.add_argument("--tasks", default="multikey,single_essay,multivalue")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budgets", default="256,512")
    ap.add_argument("--prefill-chunk", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S.local, S.rank = 64, a.rank
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="blocksparse").cuda().eval()
    budgets = [int(x) for x in a.budgets.split(",")]
    tot = {b: dict(n=0, small=0, swaps=0, k=0, hit1=0, hit2=0, margin=[]) for b in budgets}
    extra = {b: dict(cal={}, certify={}, needle_recall_by_criterion={}, mechanism={}, cover=[0, 0], needle_recall=[0, 0, 0], top1pct_recall=[0, 0, 0])
             for b in budgets}
    for task in a.tasks.split(","):
        for i in range(a.n):
            smp = make_sample(task, a.ctx, tok, 1000 + i)
            enc = tok(smp["prompt"], return_offsets_mapping=True, return_tensors="pt")
            ids = enc.input_ids.cuda()
            L = ids.shape[1]
            offs = enc.offset_mapping[0]
            nmask = torch.zeros(L + 64, dtype=torch.bool)
            for (cs, ce, tgt) in smp["spans"]:
                if tgt:
                    nmask[:L] |= (offs[:, 0] < ce) & (offs[:, 1] > cs)
            S.needle_mask = nmask
            with torch.no_grad():
                S.method, S.summ, S.W, S.C = "dense", {}, {}, {}
                cache = DynamicCache()
                cuts = list(range(0, L - 1, a.prefill_chunk)) + [L - 1]
                if len(cuts) > 2 and cuts[-1] - cuts[-2] < 2:
                    cuts.pop(-2)
                for st, en in zip(cuts[:-1], cuts[1:]):
                    model(ids[:, st:en], past_key_values=cache, use_cache=True, logits_to_keep=1)
                for b in budgets:
                    S.budget, S.diag = b, {}
                    cache.crop(-(cache.get_seq_length() - (L - 1))) if cache.get_seq_length() > L - 1 else None
                    model(ids[:, -1:], past_key_values=cache, use_cache=True, logits_to_keep=1)
                    for key, rec in S.diag.items():
                        if key in ("cal", "certify", "needle_recall_by_criterion", "mechanism"):
                            for ck, v in rec.items():
                                r = extra[b][key].setdefault(ck, [0] + [0.0] * (len(v) - 1))
                                for j in range(len(v)):
                                    r[j] += v[j]
                        elif isinstance(key, str):
                            for j in range(len(rec)):
                                extra[b][key][j] += rec[j]
                        else:
                            for k in ("n", "small", "swaps", "k", "hit1", "hit2"):
                                tot[b][k] += rec[k]
                            tot[b]["margin"] += rec["margin"]
                    S.diag = None
            del cache
            torch.cuda.empty_cache()
            print(task, i, flush=True)
    out = {}
    for b, t in tot.items():
        mg = np.array(t["margin"])
        out[b] = dict(heads=t["n"], frac_margin_lt_logB=t["small"] / t["n"],
                      margin_quantiles={q: float(np.quantile(mg, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                      swap_frac=t["swaps"] / t["k"], outlier_hit1=t["hit1"] / t["k"], outlier_hit2=t["hit2"] / t["k"],
                      logB=math.log(S.block),
                      bound_coverage=extra[b]["cover"][0] / max(1, extra[b]["cover"][1]),
                      needle_block_recall={m: extra[b]["needle_recall"][m] / max(1, extra[b]["needle_recall"][0])
                                           for m in (1, 2)},
                      top1pct_token_recall={m: extra[b]["top1pct_recall"][m] / max(1, extra[b]["top1pct_recall"][0])
                                            for m in (1, 2)},
                      needle_recall_by_criterion={k: v[0] / max(1, v[1]) for k, v in sorted(extra[b]["needle_recall_by_criterion"].items())},
                      mechanism={k: dict(oracle_mass_capture_w=v[0] / max(v[2], 1e-9), needle_block_hit_w=v[1] / max(v[2], 1e-9)) for k, v in sorted(extra[b]["mechanism"].items())},
                      certify_frac_blocks_read={k: v[1] / v[0] for k, v in sorted(extra[b]["certify"].items())},
                      calibration={k: dict(blocks=v[0], bias=v[1] / v[0], mae=v[2] / v[0])
                                   for k, v in sorted(extra[b]["cal"].items())})
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
