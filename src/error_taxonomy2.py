"""Finer error taxonomy (after review 3): for single-value tasks classify each wrong output as
correct / prefix-truncation / substitution / indel / splice-with-distractor / no-digits, record the
first wrong digit index and the number of value digits in the head block (hd)."""
import difflib
import json
import re
import sys
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tasks import make_sample  # noqa: E402


def classify(out, ans, distractors):
    m = re.search(r"\d+", out)
    if not m:
        return "no_digits", None
    got = m.group(0)
    if got == ans:
        return "correct", None
    first = next((i for i, (a, b) in enumerate(zip(got, ans)) if a != b), min(len(got), len(ans)))
    if ans.startswith(got):
        return "prefix_trunc", first
    for dv in distractors:                       # head of the answer + tail of a distractor
        for cut in range(1, len(ans)):
            if got[:cut] == ans[:cut] and got[cut:] and dv.endswith(got[cut:]) and len(got) - cut >= 2:
                return "splice_distractor", first
        if got == dv:
            return "distractor_value", 0
    if len(got) == len(ans):
        return "substitution", first
    ops = difflib.SequenceMatcher(a=ans, b=got).get_opcodes()
    return ("indel" if any(o[0] in ("insert", "delete") for o in ops) else "other"), first


def main(path, methods, B=32, model="/home/kooki3/models/qwen2.5-1.5b-instruct"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    counts = {m: Counter() for m in methods}
    rel = {m: Counter() for m in methods}
    for line in open(path):
        r = json.loads(line)
        if r["task"] == "multivalue":
            continue
        smp = make_sample(r["task"], r["ctx"], tok, r["seed"])
        enc = tok(smp["prompt"], return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        ans = r["answers"][0]
        distractors, hd, straddle = [], None, False
        for cs, ce, tgt in smp["spans"]:
            sent = smp["prompt"][cs:ce]
            if not tgt:
                distractors += re.findall(r"\d{7}", sent)
                continue
            j = sent.find(ans)
            a0, a1 = cs + j, cs + j + len(ans)
            toks = [i for i, (s, e) in enumerate(offs) if s < a1 and e > a0]
            straddle = toks[0] // B != toks[-1] // B
            hd = sum(1 for i in toks if i // B == toks[0] // B)
        for m in methods:
            if m not in r["res"]:
                continue
            kind, first = classify(r["res"][m]["out"], ans, distractors)
            counts[m][("straddle" if straddle else "inside", kind)] += 1
            if straddle and first is not None and kind not in ("correct",):
                rel[m]["first<hd" if first < hd else ("first=hd" if first == hd else "first>hd")] += 1
    for m in methods:
        print(m)
        for k, v in sorted(counts[m].items()):
            print(f"   {k[0]:8s} {k[1]:18s} {v}")
        if rel[m]:
            print("   straddled errors, first wrong digit vs head-block digit count:", dict(rel[m]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
