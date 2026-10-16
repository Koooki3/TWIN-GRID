"""Preparation manifest for official RULER data (Codex R4). Modes:
  begin OUT LEN N TASKS...   write OUT/prep_manifest.json with the expected configuration (tasks, length, N, seed,
                             tokenizer and template identity, RULER commit, source-data hashes); if OUT already has a
                             manifest with a different configuration, refuse (exit 1) — use a fresh OUT instead.
  finish OUT                 verify every task file: exactly N lines, required schema, JSON-parsable, and record each
                             file's sha256 and per-line sample_uid list (<file sha256[:16]>:<task>:<line>); exit 1 on any
                             failure.
prepare.py silently reuses an existing file whose line count equals N, so line counts alone never prove the
configuration; the begin step makes reuse safe only within an identical configuration. Further (Codex R4 rest):
  - begin refuses an OUT that already holds task data but no manifest (never adopts unmanifested data);
  - once outputs are sealed, begin and finish verify every sealed file's sha256 and UIDs and refuse on any drift
    (finish never silently re-seals changed data);
  - the configuration includes tokenizer.json, tokenizer_config.json and special_tokens_map.json (if present)."""
import hashlib
import json
import os
import subprocess
import sys

RULER = os.environ.get("RULER_ROOT", "/home/kooki3/tools/sources/ruler")
TOKENIZER = os.environ.get("RULER_TOKENIZER", "/home/kooki3/models/qwen2.5-1.5b-instruct")
TEMPLATE = os.environ.get("RULER_TEMPLATE", "qwen2.5-chat")
SEED = 42                                     # prepare.py default --random_seed
SCHEMA = {"index", "input", "outputs", "length"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def config(length, n, tasks):
    j = f"{RULER}/scripts/data/synthetic/json"
    tok_files = {f: sha(f"{TOKENIZER}/{f}") for f in ("tokenizer.json", "tokenizer_config.json",
                                                         "special_tokens_map.json") if os.path.exists(f"{TOKENIZER}/{f}")}
    return dict(tasks=tasks, max_seq_length=length, num_samples=n, random_seed=SEED, template=TEMPLATE,
                tokenizer=TOKENIZER, tokenizer_files_sha256=tok_files,
                template_py_sha256=sha(f"{RULER}/scripts/data/template.py"),
                synthetic_yaml_sha256=sha(f"{RULER}/scripts/synthetic.yaml"),
                ruler_commit=subprocess.run(["git", "-C", RULER, "rev-parse", "HEAD"], capture_output=True,
                                            text=True).stdout.strip(),
                sources={f: sha(f"{j}/{f}") for f in ("PaulGrahamEssays.json", "squad.json", "hotpotqa.json",
                                                         "english_words.json")})


def sealed_drift(out, m):
    """Problems if any sealed output file changed (sha256) since it was sealed."""
    bad = []
    for t, o in (m.get("outputs") or {}).items():
        f = os.path.join(out, t, "validation.jsonl")
        if not os.path.exists(f) or sha(f) != o["sha256"]:
            bad.append(f"{t}: sealed file missing or changed")
    return bad


def main():
    mode, out = sys.argv[1], sys.argv[2]
    path = os.path.join(out, "prep_manifest.json")
    if mode == "begin":
        cfg = config(int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:])
        os.makedirs(out, exist_ok=True)
        if os.path.exists(path):
            old = json.load(open(path))
            if {k: v for k, v in old.items() if k != "outputs"} != cfg:
                print(f"refusing: {path} has a different configuration; use a new output directory")
                sys.exit(1)
            drift = sealed_drift(out, old)
            if drift:
                print("refusing: sealed data drifted:", *drift, sep="\n  ")
                sys.exit(1)
        else:
            existing = [t for t in os.listdir(out) if os.path.exists(os.path.join(out, t, "validation.jsonl"))]
            if existing:
                print(f"refusing: {out} holds task data {existing[:3]} but no manifest; use a new output directory")
                sys.exit(1)
            json.dump(cfg, open(path, "w"), indent=1)
        print("begin ok")
        return
    m = json.load(open(path))
    drift = sealed_drift(out, m)
    if drift:
        print("finish FAILED: sealed data drifted:", *drift, sep="\n  ")
        sys.exit(1)
    bad, outs = [], dict(m.get("outputs") or {})
    for t in m["tasks"]:
        f = os.path.join(out, t, "validation.jsonl")
        if not os.path.exists(f):
            bad.append(f"{t}: missing")
            continue
        lines = open(f).read().splitlines()
        if len(lines) != m["num_samples"]:
            bad.append(f"{t}: {len(lines)} lines != {m['num_samples']}")
        for i, l in enumerate(lines):
            try:
                if not SCHEMA <= set(json.loads(l)):
                    bad.append(f"{t}: line {i} schema")
                    break
            except json.JSONDecodeError:
                bad.append(f"{t}: line {i} not JSON")
                break
        h = sha(f)
        outs[t] = dict(sha256=h, sample_uids=[f"{h[:16]}:{t}:{i}" for i in range(len(lines))])
    if bad:
        print("finish FAILED:", *bad, sep="\n  ")
        sys.exit(1)
    m["outputs"] = outs
    json.dump(m, open(path, "w"), indent=1)
    print("finish ok")


if __name__ == "__main__":
    main()
