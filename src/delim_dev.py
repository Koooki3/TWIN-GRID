"""Development runner for delimiter chunking (method delim_cobs; docs/THEORY_ADDENDUM_ZH.md). Runs src/ruler_pred.py's
main() unchanged — same data, prompt, tokenisation, chunked prefill, greedy decoding, stop set, layout (pass
--layout-unit 32), strict per-call budget assertion and output format — after importing src/delim_attn.py (which wraps
the scorer entry point) and wrapping ruler_pred's tokenizer so that, for every sample prompt, the sentence-end positions
are set (delim_attn.DELIM.ends) and per-sample delim statistics are collected. Frozen files are not modified.
Sidecars in --save_dir: delim_manifest.<task>.json (sha256 of delim_attn.py and this file, argv) and
delim_stats.<task>.jsonl (one row per sample, keyed by token_sha: delim layer-calls, fill tokens / candidate budget
mean and max, chunks per candidate token).
Usage: delim_dev.py <ruler_pred.py arguments> --methods cobs,delim_cobs"""
import atexit
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import delim_attn  # noqa: E402  (patches sparse_attn.block_scores)
import ruler_pred  # noqa: E402

argv = sys.argv[1:]
save_dir, task = argv[argv.index("--save_dir") + 1], argv[argv.index("--task") + 1]
os.makedirs(save_dir, exist_ok=True)
json.dump(dict(argv=argv, sha256={f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
                                  for f in ("delim_attn.py", "delim_dev.py", "sparse_attn.py", "ruler_pred.py")}),
          open(os.path.join(save_dir, f"delim_manifest.{task}.json"), "w"), indent=1)
stats_path = os.path.join(save_dir, f"delim_stats.{task}.jsonl")
cur = {}


def flush():
    st = delim_attn.DELIM.stats
    if cur and st and st["layer_calls"]:
        with open(stats_path, "a") as f:
            f.write(json.dumps(dict(token_sha=cur["token_sha"], layer_calls=st["layer_calls"],
                                    fill_share_mean=st["fill_sum"] / st["budget_sum"], fill_share_max=st["fill_max"],
                                    chunks_per_cand_token=st["chunks_sum"] / st["cand_tok_sum"])) + "\n")


atexit.register(flush)


class TokWrap:
    def __init__(self, tok):
        self._tok = tok

    def __getattr__(self, k):
        return getattr(self._tok, k)

    def __call__(self, *a, **k):
        r = self._tok(*a, **k)
        ids = r["input_ids"]
        if k.get("return_tensors") == "pt" and ids.shape[1] > 512:          # a sample prompt (ruler_pred line ~161)
            flush()
            flat = ids[0].tolist()
            cur["token_sha"] = hashlib.sha256(np.array(flat, dtype=np.int64).tobytes()).hexdigest()[:16]
            delim_attn.DELIM.ends = delim_attn.sentence_ends(self._tok, flat)
            delim_attn.DELIM.cache = {}
            delim_attn.DELIM.stats = dict(calls=0, fill_sum=0.0, fill_max=0.0, budget_sum=0, chunks_sum=0,
                                          cand_tok_sum=0, layer_calls=0)
        return r


_orig_from_pretrained = ruler_pred.AutoTokenizer.from_pretrained   # bound before the name is replaced


class AutoTokWrap:
    @staticmethod
    def from_pretrained(*a, **k):
        return TokWrap(_orig_from_pretrained(*a, **k))


ruler_pred.AutoTokenizer = AutoTokWrap
sys.argv = [os.path.join(HERE, "ruler_pred.py")] + argv
ruler_pred.main()
