"""Stage 1: frozen-model block-selection comparison (offline stats + end-to-end NIAH accuracy).

Per sample: dense prefill of prompt[:-1]; then for each budget one stats step (all
scorers evaluated on the same decode query), and for each (method, budget) a greedy
decode from the last prompt token with that scorer active. Resumable JSONL output.
"""
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch/CUDA init
import argparse
import random
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import sparse_attn
from sparse_attn import S
from tasks import make_sample, score, score_strict

TASK_ID = {"multikey": 1, "single_essay": 2, "single_noise": 3, "multivalue": 4}
ALL = ["o_mass", "o_peak", "o_beta2", "o_dual", "quest", "mean", "cobs", "sum_dual",
       "sk_mass", "sk_peak", "sk_beta2", "sk_dual"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/kooki3/models/qwen2.5-1.5b-instruct")
    ap.add_argument("--tasks", default="single_noise,single_essay,multikey,multivalue")
    ap.add_argument("--ctx", default="16384")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budgets", default="512,1024")
    ap.add_argument("--methods", default=",".join(ALL))
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--local", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--prefill-chunk", type=int, default=0, help="0 = one-shot prefill")
    ap.add_argument("--vram-frac", type=float, default=0.0, help="cap on this process's share of GPU memory")
    ap.add_argument("--throttle", type=float, default=0.0,
                    help="fraction of wall time left idle for other GPU users (0.5 = run at ~half speed)")
    ap.add_argument("--sparse-layers", default="", help="e.g. 2-7: apply sparse selection only in these layers")
    ap.add_argument("--w-per-head", action="store_true", help="query subspace per query head (older runs)")
    ap.add_argument("--shift-mode", default="flip", choices=["flip", "placebo"])
    ap.add_argument("--no-stats", action="store_true", help="skip the first-step offline stats (saves GPU memory)")
    ap.add_argument("--haystack", choices=["v1", "v2"], default="v1",
                    help="v2: disjoint haystack window per prompt, slot = slot0 + i * n_tasks + task position")
    ap.add_argument("--slot0", type=int, default=0, help="first v2 slot (keep dev and confirmation slots apart)")
    ap.add_argument("--seed-stream", action="store_true", help="independent seed stream per task: seed0 + 10000*task_id + i")
    ap.add_argument("--validate-only", action="store_true",
                    help="run manifest/row validation and exit (no tokenizer/model load)")
    ap.add_argument("--strict-budget", action="store_true",
                    help="fail the attempt on any per-call budget/fixed-region violation (phase 1/2)")
    ap.add_argument("--adopt-legacy", action="store_true",
                    help="development diagnostics only: resume a file that has rows but no manifest")
    ap.add_argument("--allow-code-change", action="store_true",
                    help="development diagnostics only: resume although code hashes changed (logged in manifest)")
    ap.add_argument("--shift-intervention", action="store_true",
                    help="per single-value sample, offset the selection grid so the value straddling status flips")
    ap.add_argument("--straddle-only", action="store_true",
                    help="keep only single-value samples whose value straddles a block boundary; set S.force")
    a = ap.parse_args()
    if a.sparse_layers:
        lo, hi = (int(x) for x in a.sparse_layers.split("-"))
        S.sparse_layers = set(range(lo, hi + 1))
    def pace(t_start):
        """Idle after a unit of GPU work so the desktop sharing this GPU stays responsive."""
        if a.throttle > 0:
            torch.cuda.synchronize()
            time.sleep((time.time() - t_start) * a.throttle / (1 - a.throttle))

    S.block, S.rho, S.rank, S.local, S.w_per_head = a.block, a.rho, a.rank, a.local, a.w_per_head
    S.strict_budget = a.strict_budget
    methods = a.methods.split(",")
    for m in methods:                     # wrappers use one score channel; reject union scorers under them
        base = m.split("+")[0]
        if "+" in m and (base.startswith(("sum_dual", "nf_dual", "ell_dual", "o_dual", "sk_dual"))
                         or base == "mix_cobs_mean"):
            raise SystemExit(f"unsupported combination {m}: wrappers take a single-score scorer")
        if "+ptr" in m and not (m.endswith("+ptr") and m.count("+") == 1):
            raise SystemExit(f"unsupported combination {m}: +ptr must be the only wrapper, over a single-score scorer")
    budgets = [int(x) for x in a.budgets.split(",")]
    import hashlib
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    code_ids = {f: hashlib.sha256(open(os.path.join(here, f), "rb").read()).hexdigest()[:16]
                for f in ("sparse_attn.py", "run_e2e.py", "tasks.py")}
    if a.haystack == "v2":                                   # corpus identity is part of the run identity
        from tasks import CORPUS_V2_DIR
        code_ids["corpus_v2"] = hashlib.sha256(open(f"{CORPUS_V2_DIR}/MANIFEST.sha256", "rb").read()).hexdigest()[:16]
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        code_ids[f"model:{f}"] = hashlib.sha256(open(os.path.join(a.model, f), "rb").read()).hexdigest()[:16]
    st_ = os.stat(os.path.join(a.model, "model.safetensors"))     # weights: size + mtime (no full re-read)
    code_ids["model:model.safetensors"] = f"size={st_.st_size} mtime={int(st_.st_mtime)}"
    run_id = dict(code=code_ids, argv=sys.argv[1:], model=a.model)
    DEV_FLAGS = ("--allow-code-change", "--adopt-legacy", "--validate-only")

    def norm_argv(av):
        return [x for x in av if x not in DEV_FLAGS]
    man_path = a.out + ".manifest.json"
    manifest = dict(argv=norm_argv(sys.argv[1:]), methods=methods, model=a.model, code=code_ids,
                    dev_flags=[x for x in sys.argv[1:] if x in DEV_FLAGS])
    if os.path.exists(man_path):
        old_m = json.load(open(man_path))
        if norm_argv(old_m["argv"]) != manifest["argv"] or old_m["methods"] != manifest["methods"] or old_m["model"] != a.model:
            raise SystemExit(f"refusing to resume {a.out}: argv/methods/model differ from {man_path}")
        if old_m["code"] != code_ids:
            if not a.allow_code_change:
                raise SystemExit(f"refusing to resume {a.out}: code hashes changed (pass --allow-code-change "
                                 f"for development diagnostics; never for confirmation runs)")
            old_m.setdefault("code_changes", []).append(dict(at=time.strftime("%Y-%m-%dT%H:%M:%S"), code=code_ids))
            if not a.validate_only:                              # the validation entry stays read-only
                json.dump(old_m, open(man_path, "w"), indent=1)
    else:
        legacy = sum(1 for _ in open(a.out)) if os.path.exists(a.out) else 0
        if legacy and not a.adopt_legacy:
            raise SystemExit(f"refusing to resume {a.out}: {legacy} rows but no manifest (pass --adopt-legacy "
                             f"for development diagnostics only)")
        manifest["legacy_rows_before_manifest"] = legacy
        if not a.validate_only:
            json.dump(manifest, open(man_path, "w"), indent=1)
    done = set()
    if os.path.exists(a.out):
        want = {f"{m}@{b}" for m in methods for b in a.budgets.split(",")}
        legacy_ok = (json.load(open(man_path)) if os.path.exists(man_path) else manifest).get(
            "legacy_rows_before_manifest", 0)
        lines = open(a.out).read().split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        for i, line in enumerate(lines):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(f"refusing to resume {a.out}: line {i + 1} is not valid JSON (incomplete tail?); "
                                 f"preserve the file and recover explicitly")
            key = (r["task"], r["ctx"], r["seed"])
            if key in done:
                raise SystemExit(f"refusing to resume {a.out}: duplicate row {key}")
            have = {k for k in r["res"] if not k.startswith("dense@")}
            rid = r.get("run_id")
            if i >= legacy_ok:
                man_now = old_m if os.path.exists(man_path) else manifest
                allowed_code = [man_now["code"]] + [c["code"] for c in man_now.get("code_changes", [])]
                complete = ("token_sha" in r and "spans" in r
                            and all("strict" in v and "tokens" in v for k, v in r["res"].items() if k in want))
                if (rid is None or norm_argv(rid.get("argv", [])) != manifest["argv"] or have != want
                        or rid.get("model") != a.model or rid.get("code") not in allowed_code or not complete):
                    raise SystemExit(f"refusing to resume {a.out}: row {i + 1} has a different run identity "
                                     f"(argv/methods/model/code) or lacks required fields")
            done.add(key)
    if a.validate_only:                                      # configuration/row checks only, no model load
        print(f"VALIDATION PASSED: {a.out} ({len(done)} existing rows accepted)")
        return
    if a.vram_frac:                                          # GPU configuration only after validation
        torch.cuda.set_per_process_memory_fraction(a.vram_frac)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="blocksparse").cuda().eval()
    fo = open(a.out, "a")

    for ctx in [int(x) for x in a.ctx.split(",")]:
        for task in a.tasks.split(","):
            for i in range(a.n):
                seed = a.seed0 + i + (10000 * TASK_ID[task] if a.seed_stream else 0)
                if (task, ctx, seed) in done:
                    continue
                t0 = time.time()
                tl = a.tasks.split(",")
                slot = a.slot0 + i * len(tl) + tl.index(task) if a.haystack == "v2" else None
                smp = make_sample(task, ctx, tok, seed, slot)
                enc = tok(smp["prompt"], return_offsets_mapping=True, return_tensors="pt")
                ids = enc.input_ids.cuda()
                offs = enc.offset_mapping[0]
                nmask = torch.zeros(ids.shape[1] + 64, dtype=torch.bool)
                for (cs, ce, tgt) in smp["spans"]:
                    if tgt:
                        nmask[:ids.shape[1]] |= (offs[:, 0] < ce) & (offs[:, 1] > cs)
                S.needle_mask = nmask
                spans_info = []                        # target value token spans + straddle status per grid
                for (cs, ce, tgt) in smp["spans"]:
                    if not tgt:
                        continue
                    sent = smp["prompt"][cs:ce]
                    for v in smp["answers"]:
                        j = sent.find(v)
                        if j < 0:
                            continue
                        v0, v1 = cs + j, cs + j + len(v)
                        vt = [i for i in range(ids.shape[1]) if offs[i, 0] < v1 and offs[i, 1] > v0]
                        if vt:
                            B_ = a.block
                            spans_info.append(dict(value=v, tok=[vt[0], vt[-1]],
                                                   straddle_g0=vt[0] // B_ != vt[-1] // B_,
                                                   straddle_g1=(vt[0] - B_ // 2) // B_ != (vt[-1] - B_ // 2) // B_))
                tok_sha = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()[:16]
                L = ids.shape[1]
                S.base_shift, straddle_orig = 0, None
                if a.shift_intervention:
                    if task == "multivalue":
                        continue
                    for (cs, ce, tgt) in smp["spans"]:
                        if tgt:
                            j = smp["prompt"][cs:ce].find(smp["answers"][0])
                            v0, v1 = cs + j, cs + j + len(smp["answers"][0])
                            vt = [i for i in range(L) if offs[i, 0] < v1 and offs[i, 1] > v0]
                    B_ = S.block
                    straddle_orig = vt[0] // B_ != vt[-1] // B_
                    def strad(s_):
                        return (vt[0] - s_) // B_ != (vt[-1] - s_) // B_
                    if a.shift_mode == "placebo":      # same straddling status, different grid (seeded choice)
                        opts = [s_ for s_ in range(1, B_) if strad(s_) == straddle_orig]
                        S.base_shift = random.Random(seed).choice(opts) if opts else 0
                    elif straddle_orig:                # move the grid so the value sits inside one block
                        S.base_shift = next(s_ for s_ in range(1, B_) if not strad(s_))
                    else:                              # put a block boundary in the middle of the value
                        S.base_shift = (vt[0] + len(vt) // 2) % B_
                if a.straddle_only:
                    if task == "multivalue":
                        continue
                    val_tok = []
                    for (cs, ce, tgt) in smp["spans"]:
                        if tgt:
                            j = smp["prompt"][cs:ce].find(smp["answers"][0])
                            v0, v1 = cs + j, cs + j + len(smp["answers"][0])
                            val_tok = [i for i in range(L) if offs[i, 0] < v1 and offs[i, 1] > v0]
                    if not val_tok or val_tok[0] // S.block == val_tok[-1] // S.block:
                        continue
                    hb, tb = val_tok[0] // S.block - 1, val_tok[-1] // S.block - 1     # candidate indices
                    S.force = {"head": hb, "tail": tb, "adj": hb - 1}
                max_new = 48 if task == "multivalue" else 14
                with torch.no_grad():
                    S.method, S.summ, S.W, S.C = "dense", {}, {}, {}
                    cache = DynamicCache()
                    step = a.prefill_chunk or (L - 1)
                    cuts = list(range(0, L - 1, step)) + [L - 1]
                    if len(cuts) > 2 and cuts[-1] - cuts[-2] < 2:   # a 1-token chunk would take the decode path
                        cuts.pop(-2)
                    for st, en in zip(cuts[:-1], cuts[1:]):
                        t1 = time.time()
                        model(ids[:, st:en], past_key_values=cache, use_cache=True, logits_to_keep=1)
                        pace(t1)

                    stats = {}
                    for bud in ([] if a.no_stats else budgets):
                        S.budget, S.collect, S.stat_methods = bud, {}, methods
                        cache.crop(-(cache.get_seq_length() - (L - 1))) if cache.get_seq_length() > L - 1 else None
                        t1 = time.time()
                        model(ids[:, -1:], past_key_values=cache, use_cache=True, logits_to_keep=1)
                        pace(t1)
                        stats[bud] = {m: {str(l): [v["mass"] / v["h"], v["err"] / v["h"], v["nkeep"], v["ntot"]]
                                          for l, v in d.items()} for m, d in S.collect.items()}
                        S.collect = None

                    res = {}
                    for bud in budgets:
                        for m in ["dense"] + methods:
                            if m == "dense" and bud != budgets[0]:
                                continue
                            S.method, S.budget = m, bud
                            S.tokens, S.swaps, S.ptr = {}, {}, {}   # +ptr history never crosses prompt/method
                            S.ptr_stats, S.em_viol = {}, [0, 0]     # per (prompt, budget, method) counters
                            cache.crop(-(cache.get_seq_length() - (L - 1))) if cache.get_seq_length() > L - 1 else None
                            cur = ids[:, -1:]
                            gen = []
                            for _ in range(max_new):
                                t1 = time.time()
                                lg = model(cur, past_key_values=cache, use_cache=True).logits[:, -1]
                                pace(t1)
                                nxt = lg.argmax(-1, keepdim=True)
                                gen.append(nxt.item())
                                if nxt.item() == tok.eos_token_id or nxt.item() in (tok.convert_tokens_to_ids("<|im_end|>"),):
                                    break
                                cur = nxt
                            txt = tok.decode(gen, skip_special_tokens=True)
                            tk = S.tokens.get(m)
                            res[f"{m}@{bud}"] = dict(
                                out=txt, score=score(task, txt, smp["answers"]),
                                strict=score_strict(task, txt, smp["answers"]),
                                tokens=None if not tk else dict(min=tk[0], max=tk[1], mean=tk[2] / max(tk[3], 1),
                                                               calls=tk[3], budget_violations=tk[4],
                                                               max_abs_budget_error=tk[5], fixed_region_violations=tk[6],
                                                               budget_checked_calls=tk[7]),
                                gen_steps=len(gen),
                                ptr=S.ptr_stats.get(m), em_viol=list(S.em_viol) if m.startswith("em") else None,
                                swaps=S.swaps.get(m))
                    del cache
                    S.summ = {}
                    torch.cuda.empty_cache()
                rec = dict(task=task, ctx=ctx, seed=seed, L=L, answers=smp["answers"], res=res, stats=stats,
                           block=S.block, rho=S.rho, rank=S.rank, local=S.local, sparse_layers=a.sparse_layers, force=S.force, base_shift=S.base_shift, straddle_orig=straddle_orig, shift_mode=a.shift_mode, run_id=run_id,
                           prompt_sha=hashlib.sha256(smp["prompt"].encode()).hexdigest()[:16],
                           token_sha=tok_sha, hay=smp["hay"], spans=spans_info, w_per_head=S.w_per_head, prefill_chunk=a.prefill_chunk, secs=round(time.time() - t0, 1))
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                acc = {k: v["score"] for k, v in res.items()}
                print(task, ctx, seed, L, f"{rec['secs']}s", "dense", acc.get(f"dense@{budgets[0]}"),
                      {k: acc[k] for k in acc if k.startswith(("o_dual", "sk_dual", "sk_mass", "sk_peak", "quest"))},
                      flush=True)


if __name__ == "__main__":
    main()
