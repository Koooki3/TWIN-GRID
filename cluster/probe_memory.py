"""Memory probe for tier selection (plan v1.4 §1; guide §4). Loads the model exactly as src/ruler_pred.py does (bf16,
attn_implementation "blocksparse"), builds a deterministic prompt of L token ids, runs the chunked dense prefill
(--prefill-chunk, as ruler_pred) and then a few decode steps for each method, and reports peak allocated / reserved GPU
memory per phase. It is a probe, not a guarantee: real prompts, longer generations (cwe/fwe up to 120 tokens) and
allocator fragmentation can add to it; leave >= 10% headroom. No data, no scoring.
Usage: probe_memory.py --model DIR --length L [--block 32] [--budget 1024] [--methods dense,cobs,cobs+g2f,...]
       [--steps 8] [--prefill-chunk 2048] [--layout-unit 32] [--json OUT]"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, DynamicCache

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from sparse_attn import S  # noqa: E402  (importing registers the "blocksparse" attention)

GIB = 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--methods", default="dense,cobs,cobs+g2f,nf_pt1,quest")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--layout-unit", type=int, default=32)
    ap.add_argument("--local", type=int, default=64)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--vram-frac", type=float, default=0.0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.vram_frac:
        torch.cuda.set_per_process_memory_fraction(a.vram_frac)
    S.block, S.rank, S.local, S.unit, S.budget, S.strict_budget = a.block, a.rank, a.local, a.layout_unit, a.budget, True
    total = torch.cuda.get_device_properties(0).total_memory
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="blocksparse").cuda().eval()
    weights = torch.cuda.memory_allocated()
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, min(vocab, 100000), (1, a.length), generator=g).cuda()
    res = dict(model=a.model, length=a.length, block=a.block, budget=a.budget, gpu=torch.cuda.get_device_name(0),
               gpu_total_gib=total / GIB, weights_gib=weights / GIB, phases={})
    L = a.length
    with torch.no_grad():
        S.method, S.summ, S.W, S.C = "dense", {}, {}, {}
        torch.cuda.reset_peak_memory_stats()
        cache = DynamicCache()
        cuts = list(range(0, L - 1, a.prefill_chunk)) + [L - 1]
        if len(cuts) > 2 and cuts[-1] - cuts[-2] < 2:
            cuts.pop(-2)
        for st, en in zip(cuts[:-1], cuts[1:]):
            model(ids[:, st:en], past_key_values=cache, use_cache=True, logits_to_keep=1)
        res["phases"]["prefill"] = dict(peak_alloc_gib=torch.cuda.max_memory_allocated() / GIB,
                                        peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB)
        for m in a.methods.split(","):
            S.method, S.summ, S.tokens, S.swaps = m, {}, {}, {}
            if cache.get_seq_length() > L - 1:
                cache.crop(L - 1)
            torch.cuda.reset_peak_memory_stats()
            cur = ids[:, -1:]
            for _ in range(a.steps):
                cur = model(cur, past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1, keepdim=True)
            res["phases"][f"decode:{m}"] = dict(peak_alloc_gib=torch.cuda.max_memory_allocated() / GIB,
                                                peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB)
    peak = max(v["peak_reserved_gib"] for v in res["phases"].values())
    res["peak_reserved_gib"] = peak
    res["fits_with_10pct_headroom"] = {f"{c}GB": peak * 1.1 <= c * 1e9 / GIB for c in (24, 48, 80)}
    print(json.dumps(res, indent=1))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
