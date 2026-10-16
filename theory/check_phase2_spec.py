"""Direct-exit-code tests of src/make_phase2_spec.py argv handling (Codex C2). The canonical argv must succeed (exit 0)
and write a SPEC with analysis hashes; each deviation must exit 1 without writing: repeated --rank, --block, --local,
--straddle-only, --sparse-layers=2-7, --w-per-head, --allow-code-change, a wrong --n, a missing --strict-budget.
Exit 1 on failure."""
import os
import subprocess
import sys
import tempfile

ROOT = __file__.rsplit("/theory/", 1)[0]
TOOL = [sys.executable, f"{ROOT}/src/make_phase2_spec.py"]
CANON = ["--ctx", "16384", "--budgets", "512", "--tasks", "multikey,single_essay", "--n", "150", "--seed0", "200000",
         "--seed-stream", "--haystack", "v2", "--slot0", "100", "--rank", "8", "--prefill-chunk", "2048", "--no-stats",
         "--vram-frac", "0.6", "--throttle", "0.3", "--strict-budget",
         "--methods", "cobs,cobs+g2f,nf_pt1,nf_pt1+g2f,quest,quest+g2f", "--out", "runs/phase2_g2f_confirm.jsonl"]


def run(argv):
    out = os.path.join(tempfile.mkdtemp(), "spec.json")
    r = subprocess.run(TOOL + [out, "--"] + argv, capture_output=True, text=True)
    return r.returncode, os.path.exists(out), r.stdout


bad = []
rc, wrote, out = run(CANON)
if rc != 0 or not wrote:
    bad.append(f"canonical argv refused: {out[-300:]}")
cases = {
    "repeated --rank 16": CANON + ["--rank", "16"],
    "--block 16": CANON + ["--block", "16"],
    "--local 32": CANON + ["--local", "32"],
    "--straddle-only": CANON + ["--straddle-only"],
    "--sparse-layers=2-7": CANON + ["--sparse-layers=2-7"],
    "--w-per-head": CANON + ["--w-per-head"],
    "--allow-code-change": CANON + ["--allow-code-change"],
    "--n 100": [x if x != "150" else "100" for x in CANON],
    "missing --strict-budget": [x for x in CANON if x != "--strict-budget"],
}
for name, argv in cases.items():
    rc, wrote, out = run(argv)
    print(f"{name:26s} rc={rc} wrote={wrote}")
    if rc != 1 or wrote:
        bad.append(f"{name}: rc {rc} wrote {wrote}")
print("PROBLEMS", bad if bad else "none")
sys.exit(1 if bad else 0)
