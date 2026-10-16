"""Write the frozen phase-2 spec (prereg v3) that phase2_gate.py checks every manifest and row against.
Run once at freeze time, BEFORE launch, from the commit to be used:
    make_phase2_spec.py SPEC.json -- <exact run_e2e.py argv after the script name>
Records: code/corpus/model identities computed exactly as run_e2e.py does, the exact argv (strict allow-list:
unknown, repeated or '='-form flags are refused), the fixed configuration, the 300 keys with their slots, the v2
corpus book index, and full SHA-256 of the analysis files (gate, analysis entry, Newcombe, this script, tasks.py,
prereg text) which phase2_gate.py / phase2_analysis.py verify before running. Exit 1 on refusal."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tasks  # noqa: E402

FIXED = dict(block=32, local=64, rank=8, w_per_head=False, prefill_chunk=2048, force=None, base_shift=0,
             budget=512, ctx=16384)
ARMS = ["cobs", "cobs+g2f", "nf_pt1", "nf_pt1+g2f", "quest", "quest+g2f"]


# Strict allow-list (Codex C2): every flag that may appear, each at most once, in "--flag value" form only.
# Experiment-defining flags must equal prereg v3; resource/output flags are free but recorded in the SPEC argv.
REQUIRED = {"--ctx": "16384", "--budgets": "512", "--tasks": "multikey,single_essay", "--n": "150",
            "--seed0": "200000", "--haystack": "v2", "--slot0": "100", "--rank": "8", "--prefill-chunk": "2048",
            "--methods": ",".join(ARMS)}
REQUIRED_SWITCHES = {"--seed-stream", "--strict-budget", "--no-stats"}
RESOURCE = {"--vram-frac", "--throttle", "--out"}
# Runner defaults that the fixed configuration relies on (bound by the run_e2e.py hash in the SPEC as well).
DEFAULTS = ['ap.add_argument("--block", type=int, default=32)', 'ap.add_argument("--local", type=int, default=64)',
            'ap.add_argument("--model", default="/home/kooki3/models/qwen2.5-1.5b-instruct")']
ANALYSIS_FILES = ["src/phase2_gate.py", "src/phase2_analysis.py", "src/newcombe.py", "src/make_phase2_spec.py",
                  "src/tasks.py", "theory/prereg_g2f_v3.md"]


def parse_strict(argv):
    """Parse '--flag value' / '--switch' tokens against the allow-list; reject unknown, repeated, '=' forms."""
    seen, vals, bad, i = set(), {}, [], 0
    while i < len(argv):
        t = argv[i]
        if not t.startswith("--") or "=" in t:
            bad.append(f"unexpected token {t!r}")
            i += 1
            continue
        if t in seen:
            bad.append(f"repeated {t}")
        seen.add(t)
        if t in REQUIRED_SWITCHES:
            i += 1
            continue
        if t not in REQUIRED and t not in RESOURCE:
            bad.append(f"flag not allowed for phase 2: {t}")
            i += 1
            continue
        if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
            bad.append(f"{t} has no value")
            i += 1
            continue
        vals[t] = argv[i + 1]
        i += 2
    bad += [f"{k}={vals.get(k)} (want {v})" for k, v in REQUIRED.items() if vals.get(k) != v]
    bad += [f"missing {f}" for f in sorted(REQUIRED_SWITCHES - seen)]
    bad += [f"missing {f}" for f in ("--out",) if f not in vals]
    return vals, bad


def main(out, argv):
    vals, bad = parse_strict(argv)
    src = open(os.path.join(HERE, "run_e2e.py")).read()
    bad += [f"runner default changed: {d}" for d in DEFAULTS if d not in src]
    if bad:
        print("launch argv does not match prereg v3: " + "; ".join(bad))
        sys.exit(1)
    model = "/home/kooki3/models/qwen2.5-1.5b-instruct"
    code = {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()[:16]
            for f in ("sparse_attn.py", "run_e2e.py", "tasks.py")}
    code["corpus_v2"] = hashlib.sha256(open(f"{tasks.CORPUS_V2_DIR}/MANIFEST.sha256", "rb").read()).hexdigest()[:16]
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        code[f"model:{f}"] = hashlib.sha256(open(os.path.join(model, f), "rb").read()).hexdigest()[:16]
    st = os.stat(os.path.join(model, "model.safetensors"))
    code["model:model.safetensors"] = f"size={st.st_size} mtime={int(st.st_mtime)}"
    tasks.v2_slots(16384)                                       # loads and verifies the corpus against its manifest
    keys = [["multikey", 210000 + i, 100 + 2 * i] for i in range(150)] + \
           [["single_essay", 220000 + i, 101 + 2 * i] for i in range(150)]
    root = os.path.dirname(HERE)
    analysis = {f: hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest() for f in ANALYSIS_FILES}
    spec = dict(prereg="theory/prereg_g2f_v3.md", argv=argv, model=model, code=code, fixed=FIXED, analysis=analysis,
                note="model.safetensors identity is size+mtime metadata, not a content hash",
                methods=["dense"] + ARMS, keys=keys, slot_chars=16384 * tasks.SLOT_CHARS_PER_TOKEN,
                books=[[n, b] for n, b in tasks._V2_BOOKS], corpus_chars=len(tasks._V2))
    json.dump(spec, open(out, "w"), indent=1)
    print(f"wrote {out}: {len(keys)} keys, {len(spec['books'])} books, code {code}")


if __name__ == "__main__":
    if "--" not in sys.argv:
        print(__doc__)
        sys.exit(1)
    i = sys.argv.index("--")
    main(sys.argv[1], sys.argv[i + 1:])
