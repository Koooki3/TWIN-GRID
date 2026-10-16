"""Per-digit mechanism diagnostics (theory/mech_design_v2.md). Development diagnostic, not confirmatory.

For each listed prompt: dense prefill of prompt[:-1] once; then
  1. dense teacher forcing (TF) with shared-state logging: all scorers evaluated on the dense state, dense
     attention mass on the target digit token stored as the reference weight;
  2. per arm: TF on the arm's own trajectory (own-state logging), then free running (FR) with own-state logging
     while the generated prefix still equals the gold continuation.
Gold continuation = the tokens that extend the prompt to prompt + " " + value + "." (checked to be a token-level
extension of the prompt). At step j the logits for y_j are recorded before y_j is fed.
Output: one JSONL row per prompt with all records; resume refused (always a new output file)."""
import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402

import sparse_attn  # noqa: E402,F401
from sparse_attn import S  # noqa: E402
from tasks import make_sample  # noqa: E402

SCORERS = ["cobs", "nf_pt1", "quest"]


def gold_continuation(tok, prompt, value):
    ids_p = tok(prompt).input_ids
    full = tok(prompt + " " + value + ".", return_offsets_mapping=True)
    ids_f, offs = full.input_ids, full.offset_mapping
    if ids_f[:len(ids_p)] != ids_p:
        raise RuntimeError("gold continuation is not a token-level extension of the prompt")
    cont = ids_f[len(ids_p):]
    base = len(prompt) + 1                                      # char offset of the value's first digit
    digit_of = []                                               # continuation step -> digit index or None
    for i in range(len(ids_p), len(ids_f)):
        s_, e_ = offs[i]
        digit_of.append(s_ - base if e_ - s_ == 1 and base <= s_ < base + len(value) else None)
    return ids_p, cont, digit_of


def first_divergence(gen, cont):
    for i, (a, b) in enumerate(zip(gen, cont)):
        if a != b:
            return i
    return None if len(gen) >= len(cont) else len(gen)


def check_identity(tok, task, seed, fz, ctx):
    """Rebuild one prompt and check it against its frozen identity (no model, no GPU). Returns a dict of what the
    diagnostic needs plus `problems` (empty when the prompt matches)."""
    smp = make_sample(task, ctx, tok, seed)
    value = smp["answers"][0]
    ids_p, cont, digit_of = gold_continuation(tok, smp["prompt"], value)
    enc = tok(smp["prompt"], return_offsets_mapping=True)
    offs = enc.offset_mapping
    src = None                                              # prompt token positions of the 7 digits
    for cs, ce, tgt in smp["spans"]:
        if tgt:
            j = smp["prompt"][cs:ce].find(value)
            v0 = cs + j
            src = [next(i for i, (s_, e_) in enumerate(offs) if s_ <= v0 + c < e_) for c in range(len(value))]
    L = len(ids_p)
    tok_sha = hashlib.sha256(np.array([ids_p], dtype=np.int64).tobytes()).hexdigest()[:16]   # run_e2e encoding
    p_sha = hashlib.sha256(smp["prompt"].encode()).hexdigest()[:16]
    problems = []
    # Codex E2: 23 historical identities have token_sha = null. Never fill them in; check prompt hash, L and the
    # answer instead and record the token hash as first recorded (token_identity distinguishes the two cases).
    token_identity = "verified" if fz.get("token_sha") is not None else "first_recorded"
    if p_sha != fz["prompt_sha"] or L != fz["L"] or (token_identity == "verified" and tok_sha != fz["token_sha"]):
        problems.append("prompt/token identity differs from the frozen identity")
    if fz.get("answers") is not None and fz["answers"] != smp["answers"]:
        problems.append("answers differ from the frozen identity")
    if [dd_ for dd_ in digit_of if dd_ is not None] != list(range(len(value))):
        problems.append("continuation digits do not map to 0..6")
    if src is None or len(src) != len(value) or any(tok.decode([ids_p[i]]) != value[c] for c, i in enumerate(src)):
        problems.append("source digit tokens do not decode to the value")
    if src is not None and fz.get("spans") and fz["spans"][0]["tok"] != [src[0], src[-1]]:
        problems.append("source span differs from the frozen span")
    return dict(smp=smp, value=value, ids_p=ids_p, cont=cont, digit_of=digit_of, offs=offs, src=src, L=L,
                tok_sha=tok_sha, p_sha=p_sha, token_identity=token_identity, problems=problems)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/kooki3/models/qwen2.5-1.5b-instruct")
    ap.add_argument("--prompts", required=True, help="JSON list of [task, seed]")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--arms", default="cobs,nf_pt1,quest,cobs+ftail")
    ap.add_argument("--vram-frac", type=float, default=0.6)
    ap.add_argument("--identities", required=True,
                    help="frozen prompt identities (JSON list with task, seed, prompt_sha, token_sha, L, spans)")
    ap.add_argument("--throttle", type=float, default=0.0)
    ap.add_argument("--logger-check", action="store_true",
                    help="also re-run free running with logging off and compare tokens/reads (implementation check)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preflight-only", action="store_true", help="check all prompt identities (CPU) and exit")
    a = ap.parse_args()
    if os.path.exists(a.out) and not a.preflight_only:
        raise SystemExit(f"{a.out} exists; diagnostics always write a new file")
    here = os.path.dirname(os.path.abspath(__file__))
    code = {f: hashlib.sha256(open(os.path.join(here, f), "rb").read()).hexdigest()[:16]
            for f in ("sparse_attn.py", "run_digit_diag.py", "tasks.py")}
    prompts = json.load(open(a.prompts))
    ident_raw = open(a.identities, "rb").read()
    idents = {(x["task"], x["seed"]): x for x in json.loads(ident_raw)}
    missing = [p for p in prompts if tuple(p) not in idents]
    if missing:
        raise SystemExit(f"no frozen identity for {missing[:3]}")
    mdir = a.model
    model_id = {f: hashlib.sha256(open(os.path.join(mdir, f), "rb").read()).hexdigest()[:16]
                for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")}
    st = os.stat(os.path.join(mdir, "model.safetensors"))
    model_id["model.safetensors"] = f"size={st.st_size} mtime={int(st.st_mtime)}"
    run_meta = dict(code=code, argv=sys.argv[1:], model=a.model, model_id=model_id,
                    identities_sha=hashlib.sha256(ident_raw).hexdigest()[:16],
                    defaults=dict(ctx=a.ctx, budget=a.budget, rank=a.rank, prefill_chunk=a.prefill_chunk,
                                  arms=a.arms, vram_frac=a.vram_frac, throttle=a.throttle, local=64, block=32))
    S.local, S.rank, S.budget, S.strict_budget = 64, a.rank, a.budget, True
    tok = AutoTokenizer.from_pretrained(a.model)
    pre = {}                                                    # identity preflight of ALL prompts before any GPU use
    for task, seed in prompts:
        pre[(task, seed)] = check_identity(tok, task, seed, idents[(task, seed)], a.ctx)
    bad = {k: v["problems"] for k, v in pre.items() if v["problems"]}
    kinds = {k: v["token_identity"] for k, v in pre.items()}
    print(f"preflight: {len(pre)} prompts, {len(bad)} with problems; token identity "
          f"verified {sum(x == 'verified' for x in kinds.values())}, first recorded "
          f"{sum(x == 'first_recorded' for x in kinds.values())}", flush=True)
    if bad:
        raise SystemExit("preflight failed: " + "; ".join(f"{k[0]} {k[1]}: {', '.join(v)}" for k, v in bad.items()))
    if a.preflight_only:
        return
    if a.vram_frac:
        torch.cuda.set_per_process_memory_fraction(a.vram_frac)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="blocksparse").cuda().eval()
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    fo = open(a.out, "w")
    for task, seed in prompts:
        t0 = time.time()
        c = pre[(task, seed)]
        smp, value, ids_p, cont, digit_of = c["smp"], c["value"], c["ids_p"], c["cont"], c["digit_of"]
        offs, src, L, tok_sha, p_sha, token_identity = c["offs"], c["src"], c["L"], c["tok_sha"], c["p_sha"], c["token_identity"]
        key_pos = None                                          # first token of the key word in the target needle
        for cs, ce, tgt in smp["spans"]:
            if tgt:
                kpos = cs + smp["prompt"][cs:ce].find(" for ") + len(" for ")
                key_pos = next(i for i, (s_, e_) in enumerate(offs) if s_ <= kpos < e_)
        ids = torch.tensor([ids_p], device="cuda")
        targets = [None if dj is None else src[dj] for dj in digit_of]
        row = dict(task=task, seed=seed, ctx=a.ctx, L=L, value=value, src_tokens=src, key_pos=key_pos, cont=cont,
                   digit_of=digit_of, prompt_sha=p_sha, token_sha=tok_sha, identity_checked=True,
                   token_identity=token_identity,
                   straddle=src[0] // 32 != src[-1] // 32, run=run_meta, arms={})
        probe_pos = dict(head=src[0], tail=src[-1], key=key_pos)

        def pace(t_start):
            if a.throttle > 0:
                torch.cuda.synchronize()
                time.sleep((time.time() - t_start) * a.throttle / (1 - a.throttle))
        with torch.no_grad():
            S.method, S.summ, S.W, S.C, S.dd = "dense", {}, {}, {}, None
            cache = DynamicCache()
            for st in range(0, L - 1, a.prefill_chunk):
                model(ids[:, st:min(st + a.prefill_chunk, L - 1)], past_key_values=cache, use_cache=True,
                      logits_to_keep=1)
            weights = {}

            def tf(method, mode):
                S.method, S.tokens, S.swaps, S.ptr, S.ptr_stats = method, {}, {}, {}, {}
                cache.crop(-(cache.get_seq_length() - (L - 1))) if cache.get_seq_length() > L - 1 else None
                S.dd = dict(mode=mode, scorers=SCORERS, out=[], weights=weights, step=0, target_pos=None,
                            frozen_len=L - 1, probe_pos=probe_pos, swap_log=[])
                cur, steps = ids[:, -1:], []
                for j, y in enumerate(cont):
                    S.dd["step"], S.dd["target_pos"] = j, targets[j]
                    t1 = time.time()
                    lp = model(cur, past_key_values=cache, use_cache=True).logits[0, -1].float().log_softmax(-1)
                    pace(t1)
                    top2 = lp.topk(2)
                    best_other = top2.values[1] if int(top2.indices[0]) == y else top2.values[0]
                    steps.append(dict(j=j, gold=y, gold_logprob=float(lp[y]), margin=float(lp[y] - best_other),
                                      argmax=int(top2.indices[0])))
                    cur = torch.tensor([[y]], device="cuda")
                out = dict(state_origin="dense_TF" if mode == "shared" else "arm_TF", steps=steps,
                           records=S.dd["out"], swap_log=S.dd["swap_log"], swaps=dict(S.swaps), tokens=dict(S.tokens))
                S.dd = None
                return out

            def fr(method, log=True):
                S.method, S.tokens, S.swaps, S.ptr, S.ptr_stats = method, {}, {}, {}, {}
                cache.crop(-(cache.get_seq_length() - (L - 1))) if cache.get_seq_length() > L - 1 else None
                S.dd = (dict(mode="own", scorers=SCORERS, out=[], weights=weights, step=0, target_pos=None,
                             frozen_len=L - 1, probe_pos=probe_pos, swap_log=[]) if log else None)
                cur, gen, steps = ids[:, -1:], [], []
                for j in range(14):
                    same = gen == cont[:j] and j < len(cont)
                    if S.dd is not None:
                        S.dd["step"], S.dd["target_pos"] = j, (targets[j] if same else None)
                    t1 = time.time()
                    lp = model(cur, past_key_values=cache, use_cache=True).logits[0, -1].float().log_softmax(-1)
                    pace(t1)
                    nxt = int(lp.argmax())
                    if same:
                        y = cont[j]
                        top2 = lp.topk(2)
                        best_other = top2.values[1] if int(top2.indices[0]) == y else top2.values[0]
                        steps.append(dict(j=j, gold=y, gold_logprob=float(lp[y]), margin=float(lp[y] - best_other),
                                          argmax=nxt))
                    gen.append(nxt)
                    if nxt in eos:
                        break
                    cur = torch.tensor([[nxt]], device="cuda")
                out = dict(state_origin="arm_FR", gen=gen, text=tok.decode(gen, skip_special_tokens=True),
                           first_divergence=first_divergence(gen, cont), steps=steps,
                           records=S.dd["out"] if S.dd is not None else None,
                           swap_log=S.dd["swap_log"] if S.dd is not None else None,
                           swaps=dict(S.swaps), tokens=dict(S.tokens))
                S.dd = None
                return out

            row["dense_tf"] = tf("dense", "shared")
            for arm in a.arms.split(","):
                S.force = None
                if "+f" in arm:
                    hb, tb = src[0] // 32 - 1, src[-1] // 32 - 1
                    S.force = {"head": hb, "tail": tb, "adj": hb - 1}
                row["arms"][arm] = dict(tf=tf(arm, "own"), fr=fr(arm), force=S.force)
                if a.logger_check:                              # logging must not change generation or reads
                    off = fr(arm, log=False)
                    on = row["arms"][arm]["fr"]
                    row["arms"][arm]["logger_check"] = dict(
                        same_tokens=off["gen"] == on["gen"], same_reads=off["tokens"] == on["tokens"],
                        same_swaps=off["swaps"] == on["swaps"])
            S.force = None
            del cache
            torch.cuda.empty_cache()
        row["secs"] = round(time.time() - t0, 1)
        fo.write(json.dumps(row) + "\n")
        fo.flush()
        print(task, seed, row["straddle"], row["secs"], flush=True)


if __name__ == "__main__":
    main()
