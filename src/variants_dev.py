"""Development runner for the simple-alternative and summary-type variants (reviewer batch 7; user follow-up): runs
src/ruler_pred.py's main() unchanged after importing src/nbh_attn.py (adds cobs+nbh by wrapping the scorer entry point;
cobs+nb, cobs+nbr, mean, mean+g2f and quest already exist in the frozen sparse_attn.py). Frozen files are not modified.
Writes delim-style sidecar variants_manifest.<task>.json (sha256 of nbh_attn.py, this file, sparse_attn.py,
ruler_pred.py; argv). Usage: variants_dev.py <ruler_pred.py arguments>"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nbh_attn  # noqa: E402,F401  (patches sparse_attn.block_scores)
import ruler_pred  # noqa: E402

argv = sys.argv[1:]
save_dir, task = argv[argv.index("--save_dir") + 1], argv[argv.index("--task") + 1]
os.makedirs(save_dir, exist_ok=True)
json.dump(dict(argv=argv, sha256={f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
                                  for f in ("nbh_attn.py", "variants_dev.py", "sparse_attn.py", "ruler_pred.py")}),
          open(os.path.join(save_dir, f"variants_manifest.{task}.{argv[argv.index('--methods') + 1]}.json"), "w"),
          indent=1)
sys.argv = [os.path.join(HERE, "ruler_pred.py")] + argv
ruler_pred.main()
