"""Write the EXPECTED run identity of the 63-prompt mechanism run (Codex N1-final-b) from the launch command in
scripts/queue_mech_full.sh and the frozen source/model/identity files — BEFORE that run starts, never from its output.
src/mech_matrix.py compares every row's run identity with this file in full mode.
Usage: make_mech_expected.py OUT.json"""
import hashlib
import json
import os
import sys

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDS = "/home/kooki3/docs/projects/attention-review/2026-10-16/hourly-mech-v2-1426/original-prompt-identities.json"
MODEL = "/home/kooki3/models/qwen2.5-1.5b-instruct"


def h(p, n=None):
    d = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return d[:n] if n else d


def main(out):
    code = {f: h(f"{WT}/src/{f}", 16) for f in ("sparse_attn.py", "run_digit_diag.py", "tasks.py")}
    mid = {f: h(f"{MODEL}/{f}", 16) for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
                                              "generation_config.json")}
    st = os.stat(f"{MODEL}/model.safetensors")
    mid["model.safetensors"] = f"size={st.st_size} mtime={int(st.st_mtime)}"
    argv = ["--prompts", f"{WT}/results/mech_prompts_full.json", "--identities", IDS, "--vram-frac", "0.6",
            "--throttle", "0.3", "--logger-check", "--out", f"{WT}/runs/mech_full_16k.jsonl"]
    launch = open(f"{WT}/scripts/queue_mech_full.sh").read()
    for tok in ("mech_prompts_full.json", "--vram-frac 0.6 --throttle 0.3 --logger-check", "mech_full_16k.jsonl"):
        if tok not in launch:
            print(f"launch script does not contain {tok!r}; refusing")
            sys.exit(1)
    run = dict(code=code, argv=argv, model=MODEL, model_id=mid, identities_sha=h(IDS, 16),
               defaults=dict(ctx=16384, budget=512, rank=8, prefill_chunk=2048, arms="cobs,nf_pt1,quest,cobs+ftail",
                             vram_frac=0.6, throttle=0.3, local=64, block=32))
    json.dump(dict(note="EXPECTED identity written before the run (not derived from its output)", run=run,
                   prompts_sha256=h(f"{WT}/results/mech_prompts_full.json"), identities_sha256=h(IDS),
                   launch_script_sha256=h(f"{WT}/scripts/queue_mech_full.sh")), open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
