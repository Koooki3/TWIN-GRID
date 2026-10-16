"""Exploratory paired comparison of new arms against phase-1 arms on the same development prompts (strict scoring).
Checks that every shared key has the same prompt and token hashes; reports accuracy and improve/worsen counts.
Usage: dev_compare.py NEW.jsonl [PHASE1.jsonl]   (descriptive only, no significance claims)"""
import json
import sys
from collections import Counter


def rows(path):
    return {(r["task"], r["seed"]): r for r in map(json.loads, open(path))}


def st(r, m):
    return int(r["res"][m]["strict"])


def main(new_path, ref_path):
    new, ref = rows(new_path), rows(ref_path)
    keys = sorted(set(new) & set(ref))
    bad = [k for k in keys if (new[k]["prompt_sha"], new[k]["token_sha"]) != (ref[k]["prompt_sha"], ref[k]["token_sha"])]
    if bad:
        sys.exit(f"identity mismatch on {len(bad)} keys, e.g. {bad[:3]}")
    arms_new = sorted({m for k in keys for m in new[k]["res"] if not m.startswith("dense")})
    arms_ref = sorted({m for k in keys for m in ref[k]["res"]})
    print(f"shared prompts {len(keys)} (new file {len(new)}, reference {len(ref)})")
    for m in arms_ref + arms_new:
        src = new if m in arms_new else ref
        by = Counter(k[0] for k in keys if st(src[k], m))
        print(f"  {m:16s} {sum(by.values()):3d}/{len(keys)}  {dict(by)}")
    for m in arms_new:
        name, bud = m.split("@")
        base = name.split("+")[0]
        base = "quest" if base == "em" else base           # em is compared with the box scorer it refines
        for r_ in sorted({f"{base}@{bud}", f"{base}+g2f@{bud}", f"quest@{bud}", f"quest+g2f@{bud}"} & set(arms_ref)):
            c = Counter((st(new[k], m), st(ref[k], r_)) for k in keys)
            print(f"  {m} vs {r_}: better {c[(1, 0)]} worse {c[(0, 1)]} both right {c[(1, 1)]} both wrong {c[(0, 0)]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else
         __file__.rsplit("/src/", 1)[0] + "/runs/phase1_g2f_dev.jsonl")
