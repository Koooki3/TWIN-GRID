"""Write the analysis manifest for src/mech_matrix.py (Codex N1). Run AFTER the mechanism run finishes and BEFORE any
analysis; commit the manifest, then run the analysis with --manifest. Records the single run identity found in the
JSONL (refuses if rows disagree), the sha256 of the JSONL, the prompt list and the identity file, and the sha256 of
the analysis script. Usage: make_mech_manifest.py MECH.jsonl PROMPTS.json IDENTITIES.json OUT.json"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(mech, prompts, idents, out):
    runs = {json.dumps(json.loads(l).get("run"), sort_keys=True) for l in open(mech)}
    if len(runs) != 1:
        print(f"refusing: {len(runs)} distinct run identities in {mech}")
        sys.exit(1)
    m = dict(run=json.loads(next(iter(runs))), mech_sha256=sha(mech), prompts_sha256=sha(prompts),
             identities_sha256=sha(idents), mech_matrix_sha256=sha(os.path.join(HERE, "mech_matrix.py")))
    json.dump(m, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:5])
