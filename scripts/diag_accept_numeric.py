"""Acceptance diagnostic (plan §3.2: keep the raw result, explain a backend difference, do not tune): for one frozen
sample, greedy-decode the same prompt tokens with plain HF sdpa attention (the official wrapper's fallback path) in two
ways — (A) one-shot prefill, (B) chunked prefill as in src/ruler_pred.py (--prefill-chunk 2048) — and report, per path,
the first step where A and B pick different tokens and the top-1 minus top-2 logit margin there.
Usage: diag_accept_numeric.py MODEL_DIR DATA_JSONL LINE MAX_NEW [CHUNK]"""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

mdir, data, line, max_new = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
chunk = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
torch.cuda.set_per_process_memory_fraction(0.6)
smp = [json.loads(l) for l in open(data)][line]
tok = AutoTokenizer.from_pretrained(mdir)
model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
ids = tok(smp["input"] + smp.get("answer_prefix", ""), return_tensors="pt").input_ids.cuda()
L = ids.shape[1]
eos = set([model.generation_config.eos_token_id] if isinstance(model.generation_config.eos_token_id, int)
          else model.generation_config.eos_token_id)


@torch.no_grad()
def run(step):
    cache = DynamicCache()
    cuts = list(range(0, L - 1, step)) + [L - 1] if step else [0, L - 1]
    if len(cuts) > 2 and cuts[-1] - cuts[-2] < 2:
        cuts.pop(-2)
    for st, en in zip(cuts[:-1], cuts[1:]):
        model(ids[:, st:en], past_key_values=cache, use_cache=True, logits_to_keep=1)
    cur, gen, margins, top2 = ids[:, -1:], [], [], []
    for _ in range(max_new):
        lg = model(cur, past_key_values=cache, use_cache=True).logits[:, -1].float()
        top = lg.topk(2, dim=-1)
        margins.append((top.values[0, 0] - top.values[0, 1]).item())
        top2.append((top.indices[0, 0].item(), top.indices[0, 1].item()))
        nxt = top.indices[:, :1]
        gen.append(nxt.item())
        if nxt.item() in eos:
            break
        cur = nxt
    return gen, margins, top2


gA, mA, tA = run(0)
gB, mB, tB = run(chunk)
k = next((i for i, (x, y) in enumerate(zip(gA, gB)) if x != y), None)
print(f"L={L}; A one-shot prefill: {tok.decode(gA, skip_special_tokens=True)!r}")
print(f"B chunked {chunk}: {tok.decode(gB, skip_special_tokens=True)!r}")
if k is None:
    print("A and B identical; min margin A", min(mA), "B", min(mB))
else:
    print(f"first divergence at step {k}: A picks {tok.decode([gA[k]])!r} (margin {mA[k]:.4f}), "
          f"B picks {tok.decode([gB[k]])!r} (margin {mB[k]:.4f}); margins before divergence min "
          f"A {min(mA[:k] or [float('nan')]):.4f} B {min(mB[:k] or [float('nan')]):.4f}")
for name, g, m, t in (("A", gA, mA, tA), ("B", gB, mB, tB)):
    i = min(range(len(m)), key=lambda j: m[j])
    print(f"{name}: smallest margin {m[i]:.4f} at step {i} after {tok.decode(g[:i])!r}: top-2 "
          f"{tok.decode([t[i][0]])!r} vs {tok.decode([t[i][1]])!r}")
