"""Where does the block boundary cut the value? For naturally crossing prompts (original grid) and for the
flip arm's intervened cuts, list value tokens on the head side vs tail side and each scorer's outcome.
Descriptive only. Usage: cut_position.py SHIFT_FLIP.jsonl UNSHIFTED.jsonl"""
import json
import sys


def main(flip, unshifted, B=32, methods=("cobs", "nf_pt1", "quest")):
    base = {(r["task"], r["seed"]): r for r in map(json.loads, open(unshifted))}
    print("| arm | task | seed | value tokens head/tail | " + " | ".join(methods) + " |")
    print("|---|---|---|---|" + "---|" * len(methods))
    for r in map(json.loads, open(flip)):
        sp = r["spans"][0]
        t0, t1 = sp["tok"]
        L = t1 - t0 + 1
        b = base[(r["task"], r["seed"])]
        if r["straddle_orig"]:                       # natural cut on the original grid
            head = (t0 // B + 1) * B - t0
            out = " | ".join(str(int(b["res"][f"{m}@512"]["score"])) for m in methods)
            print(f"| natural | {r['task']} | {r['seed']} | {head}/{L - head} | {out} |")
        else:                                        # intervened cut: grid offset = base_shift
            sh = r["base_shift"]
            head = ((t0 - sh) // B + 1) * B + sh - t0
            out = " | ".join(str(int(r["res"][f"{m}@512"]["score"])) for m in methods)
            print(f"| flip-cut | {r['task']} | {r['seed']} | {head}/{L - head} | {out} |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
