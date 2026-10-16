"""Official RULER (v1 scripts, commit c3f5e3b) prediction step with this project's block-sparse decode attention.
Replaces only `pred/call_api.py`; RULER's `data/prepare.py` and `eval/evaluate.py` are used unchanged.

Protocol mirrored from the official HF path (pred/model_wrappers.py HuggingFaceModel, call_api.py):
  prompt = sample["input"] + sample.get("answer_prefix", "")   (the model template was applied by prepare.py);
  tokenisation with the tokenizer's default special-token handling (official behaviour; for Llama-3 this repeats the
  BOS already in the template — recorded, not "fixed", so dense matches the official path; --no-special-tokens
  switches it off for a separately labelled variant);
  greedy decoding, max_new_tokens = tokens_to_generate of the task (synthetic.yaml task -> constants.TASKS), stop at
  EOS; decoded with skip_special_tokens; the official stop-word split is applied (default: none).
Stop set: the model's generation_config eos_token_id (int or list) plus the tokenizer's eos_token_id, recorded in the
manifest (Codex R3). Greedy argmax decoding = do_sample False, repetition_penalty 1 (the official call_api values with
temperature 0).
Output: one jsonl per (task, method@budget) in --save_dir/<method@budget>/<task>.jsonl with the official prediction
fields {index, pred, input, outputs, others, truncation, length} — `input` is the ACTUAL generation prompt
(input + answer_prefix), the original prepare.py input is kept as raw_input — plus audit fields {sample_uid, input_sha,
token_sha, n_input_tokens, tokens (per-call budget counters), swaps}.
Sample identity (Codex R2): the official NIAH `index` is a character position, not a unique id; this script uses
sample_uid = "<data-file sha256[:16]>:<task>:<0-based line number>" for resume, pairing and completeness; the
official index is kept for traceability. "First 100 samples" means line numbers 0..99 of the frozen file.
Manifest (Codex R4): <save_dir>/manifest.<task>.json holds code hashes, argv, model files (incl. generation_config and
weights size+mtime), RULER commit, tokens_to_generate, stop set, data-file hash; it must match on resume, and existing
prediction files without a manifest are refused (never adopted).
Per-arm state (Codex R1): block summaries (S.summ) are cleared at the start of every arm, so cached summaries never
cross arms whose generated tokens differ. Each row records n_gen and gen_in_candidates: whether, at the last decode
step, the full candidate blocks of grid 0 reached beyond the prompt (i.e. generated tokens were inside a selectable
block) — only then can a cross-arm summary reuse show up at all (Codex A2).
Model identity (Codex A3): config/tokenizer/generation_config hashes plus either model.safetensors (size+mtime) or,
for sharded checkpoints, model.safetensors.index.json (sha256) and every shard it references (size+mtime); a
directory with neither is refused. --model-revision (HF commit) is recorded when given.
Decode attention: prefill is dense (chunked); decode steps use the selected method with the per-call budget
invariant (--strict-budget). Accuracy only — the mask is simulated, no speed claim.
Usage: ruler_pred.py --ruler ROOT --data_dir DIR --save_dir DIR --task T --methods dense,quest,... --budget 1024
       [--block 32] [--rank 8] [--limit N] [--vram-frac F] [--throttle T]"""
import argparse
import glob
import hashlib
import subprocess
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sparse_attn  # noqa: E402,F401  (registers the "blocksparse" attention)
from sparse_attn import S  # noqa: E402


def model_identity(model_dir, revision=""):
    """Model identity for the manifest (single-file or sharded safetensors); refuses an unrecognised layout."""
    ident = {}
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        ident[f"model:{f}"] = hashlib.sha256(open(os.path.join(model_dir, f), "rb").read()).hexdigest()[:16]
    single = os.path.join(model_dir, "model.safetensors")
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(single):
        st_ = os.stat(single)
        ident["model:model.safetensors"] = f"size={st_.st_size} mtime={int(st_.st_mtime)}"
    elif os.path.exists(index):
        ident["model:model.safetensors.index.json"] = hashlib.sha256(open(index, "rb").read()).hexdigest()[:16]
        for shard in sorted(set(json.load(open(index))["weight_map"].values())):
            st_ = os.stat(os.path.join(model_dir, shard))          # raises if a referenced shard is missing
            ident[f"model:{shard}"] = f"size={st_.st_size} mtime={int(st_.st_mtime)}"
    else:
        raise SystemExit(f"refusing: {model_dir} has neither model.safetensors nor model.safetensors.index.json")
    ident["model:revision"] = revision or "unrecorded"
    return ident


def tokens_to_generate(ruler_root, task):
    cfg = yaml.safe_load(open(os.path.join(ruler_root, "scripts", "synthetic.yaml")))[task]
    sys.path.insert(0, os.path.join(ruler_root, "scripts", "data", "synthetic"))
    from constants import TASKS
    return TASKS[cfg["task"]]["tokens_to_generate"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ruler", default="/home/kooki3/tools/sources/ruler")
    ap.add_argument("--model", default="/home/kooki3/models/qwen2.5-1.5b-instruct")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--methods", default="dense")
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--local", type=int, default=64)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--layout-unit", type=int, default=32,
                    help="sink / tail / candidate-budget grid (plan v1.4 G3 matching; = block 32 is the original layout)")
    ap.add_argument("--limit", type=int, default=0, help="first N samples only (smoke)")
    ap.add_argument("--stop_words", default="")
    ap.add_argument("--no-special-tokens", action="store_true")
    ap.add_argument("--strict-budget", action="store_true")
    ap.add_argument("--model-revision", default="", help="HF commit of the model snapshot (recorded in the manifest)")
    ap.add_argument("--vram-frac", type=float, default=0.0)
    ap.add_argument("--throttle", type=float, default=0.0)
    a = ap.parse_args()

    code = {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()[:16]
            for f in ("sparse_attn.py", "ruler_pred.py")}
    code.update(model_identity(a.model, a.model_revision))
    data_file = os.path.join(a.data_dir, a.task, "validation.jsonl")
    data_sha = hashlib.sha256(open(data_file, "rb").read()).hexdigest()
    code["data"] = data_sha[:16]
    ruler_commit = subprocess.run(["git", "-C", a.ruler, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    gen_cfg = json.load(open(os.path.join(a.model, "generation_config.json")))
    tok_eos = AutoTokenizer.from_pretrained(a.model).eos_token_id
    cfg_eos = gen_cfg.get("eos_token_id")
    stop_ids = sorted({tok_eos} | set(cfg_eos if isinstance(cfg_eos, list) else [cfg_eos]) - {None})
    max_new = tokens_to_generate(a.ruler, a.task)
    os.makedirs(a.save_dir, exist_ok=True)
    man_path = os.path.join(a.save_dir, f"manifest.{a.task}.json")
    manifest = dict(code=code, argv=sys.argv[1:], model=a.model, ruler_commit=ruler_commit, tokens_to_generate=max_new,
                    stop_ids=stop_ids, special_tokens=not a.no_special_tokens)
    existing = [p for p in glob.glob(os.path.join(a.save_dir, "*", f"{a.task}.jsonl")) if os.path.getsize(p) > 0]
    if os.path.exists(man_path):
        if json.load(open(man_path)) != manifest:
            raise SystemExit(f"refusing to resume: {man_path} differs (code/argv/model/data/protocol changed)")
    elif existing:
        raise SystemExit(f"refusing: prediction files {existing[:2]} exist without {man_path}; not adopting them")
    json.dump(manifest, open(man_path, "w"), indent=1)

    S.block, S.rank, S.local, S.strict_budget = a.block, a.rank, a.local, a.strict_budget
    S.unit = a.layout_unit
    if a.vram_frac:
        torch.cuda.set_per_process_memory_fraction(a.vram_frac)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation="blocksparse").cuda().eval()
    stops = [w for w in a.stop_words.split(",") if w]
    eos = set(stop_ids)
    data = [json.loads(l) for l in open(data_file)]
    for i, smp in enumerate(data):
        smp["_uid"] = f"{data_sha[:16]}:{a.task}:{i}"
    if a.limit:
        data = data[:a.limit]
    methods = a.methods.split(",")
    outs, done = {}, {}
    for m in methods:
        d = os.path.join(a.save_dir, f"{m}@{a.budget}")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{a.task}.jsonl")
        done[m] = {json.loads(l)["sample_uid"] for l in open(p)} if os.path.exists(p) else set()
        outs[m] = open(p, "a")

    def pace(t0):
        if a.throttle > 0:
            torch.cuda.synchronize()
            time.sleep((time.time() - t0) * a.throttle / (1 - a.throttle))

    for smp in data:
        todo = [m for m in methods if smp["_uid"] not in done[m]]
        if not todo:
            continue
        prompt = smp["input"] + smp.get("answer_prefix", "")
        ids = tok(prompt, return_tensors="pt", add_special_tokens=not a.no_special_tokens).input_ids.cuda()
        L = ids.shape[1]
        tsha = hashlib.sha256(ids.cpu().numpy().astype(np.int64).tobytes()).hexdigest()[:16]
        with torch.no_grad():
            S.method, S.summ, S.W, S.C = "dense", {}, {}, {}
            cache = DynamicCache()
            step = a.prefill_chunk or (L - 1)
            cuts = list(range(0, L - 1, step)) + [L - 1]
            if len(cuts) > 2 and cuts[-1] - cuts[-2] < 2:
                cuts.pop(-2)
            for st, en in zip(cuts[:-1], cuts[1:]):
                t0 = time.time()
                model(ids[:, st:en], past_key_values=cache, use_cache=True, logits_to_keep=1)
                pace(t0)
            for m in todo:
                S.method, S.budget = m, a.budget
                S.tokens, S.swaps, S.ptr, S.ptr_stats, S.em_viol = {}, {}, {}, {}, [0, 0]
                S.summ = {}                                  # R1: never reuse summaries across arms
                if cache.get_seq_length() > L - 1:
                    cache.crop(L - 1)
                cur, gen = ids[:, -1:], []
                for _ in range(max_new):
                    t0 = time.time()
                    lg = model(cur, past_key_values=cache, use_cache=True).logits[:, -1]
                    pace(t0)
                    nxt = lg.argmax(-1, keepdim=True)
                    gen.append(nxt.item())
                    if nxt.item() in eos:
                        break
                    cur = nxt
                text = tok.decode(gen, skip_special_tokens=True)
                sh_saved = (S.shift, S.base_shift)
                S.shift, S.base_shift = 0, 0
                n_last = L + len(gen) - 1                    # keys seen at the last decode call
                _, _, c1_last = sparse_attn._layout(n_last)
                S.shift, S.base_shift = sh_saved
                gen_in_cand = bool(c1_last > L)
                for w in stops:
                    text = text.split(w)[0]
                tk = S.tokens.get(m)
                row = dict(index=smp["index"], pred=text, input=prompt, outputs=smp["outputs"],
                           others=smp.get("others", {}), truncation=smp.get("truncation", -1),
                           length=smp.get("length", -1), sample_uid=smp["_uid"], raw_input=smp["input"],
                           input_sha=hashlib.sha256(prompt.encode()).hexdigest()[:16], token_sha=tsha, n_input_tokens=L,
                           tokens=None if not tk else dict(calls=tk[3], budget_violations=tk[4], max_abs_budget_error=tk[5],
                                                           fixed_region_violations=tk[6], budget_checked_calls=tk[7]),
                           swaps=S.swaps.get(m), n_gen=len(gen), gen_in_candidates=gen_in_cand)
                outs[m].write(json.dumps(row) + "\n")
                outs[m].flush()
            del cache
            S.summ = {}
            torch.cuda.empty_cache()
        print(a.task, smp["_uid"], L, {m: "ok" for m in todo}, flush=True)


if __name__ == "__main__":
    main()
