"""bf16 tie diagnostic for one same-input acceptance mismatch (plan v1.4 §3.2; reviewer batch 3 conditions a-d).
Greedy-decodes the frozen prompt on the official HF fallback path (AutoModelForCausalLM, bf16, sdpa, one-shot prefill)
and compares, step by step, with the adapter's dense prediction text for the same sample:
  - input tokens: the adapter row's token_sha must equal the sha of the reference input ids (same tokenisation);
  - first divergence step k = first step where the reference prefix (decoded) stops being a prefix of the adapter text;
  - swap: the adapter continues with the reference's SECOND choice at step k (decode(ref[:k] + [2nd]) is a prefix of
    the adapter text) — condition c;
  - margin: reference top-1 minus top-2 logit at step k <= 2 * ulp_bf16(max(|l1|, |l2|)), ulp_bf16(x) =
    2**(floor(log2 x) - 7) — condition b.
tie = input equal and swap and margin condition. The JSON is kept for every mismatch (condition d).
Usage: accept_tie_diag.py MODEL_DIR DATA_JSONL LINE ADAPTER_PRED_JSONL MAX_NEW OUT.json"""
import hashlib
import json
import math
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def ulp_bf16(x):
    return 2.0 ** (math.floor(math.log2(abs(x))) - 7) if x else 2.0 ** -133


def main(mdir, data, line, pred_path, max_new, out):
    smp = [json.loads(l) for l in open(data)][line]
    sha = hashlib.sha256(open(data, "rb").read()).hexdigest()
    task = data.rstrip("/").split("/")[-2]
    uid = f"{sha[:16]}:{task}:{line}"
    row = next(json.loads(l) for l in open(pred_path) if json.loads(l)["sample_uid"] == uid)
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    ids = tok(smp["input"] + smp.get("answer_prefix", ""), return_tensors="pt").input_ids.cuda()
    tsha = hashlib.sha256(ids.cpu().numpy().astype(np.int64).tobytes()).hexdigest()[:16]
    eos = model.generation_config.eos_token_id
    eos = set([eos] if isinstance(eos, int) else eos)
    target = row["pred"]
    gen, res = [], dict(uid=uid, input_equal=tsha == row.get("token_sha"), divergence_step=None)
    with torch.no_grad():
        out_ = model(ids, use_cache=True, logits_to_keep=1)
        cache, lg = out_.past_key_values, out_.logits[:, -1].float()
        for k in range(max_new):
            top = lg.topk(2, dim=-1)
            t1, t2 = top.indices[0, 0].item(), top.indices[0, 1].item()
            l1, l2 = top.values[0, 0].item(), top.values[0, 1].item()
            if not target.startswith(tok.decode(gen + [t1], skip_special_tokens=True)) and t1 not in eos:
                res.update(divergence_step=k, ref_top2=[tok.decode([t1]), tok.decode([t2])], ref_logits=[l1, l2],
                           margin=l1 - l2, ulp=ulp_bf16(max(abs(l1), abs(l2))),
                           swap=target.startswith(tok.decode(gen + [t2], skip_special_tokens=True)))
                break
            if t1 in eos:
                break
            gen.append(t1)
            o = model(torch.tensor([[t1]], device=ids.device), past_key_values=cache, use_cache=True)
            cache, lg = o.past_key_values, o.logits[:, -1].float()
    res["tie"] = bool(res["input_equal"] and res.get("swap") and res.get("margin") is not None
                      and res["margin"] <= 2 * res["ulp"])
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6])
