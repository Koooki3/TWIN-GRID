"""Classify wrong answers: distractor value, near-miss digits, other number, no number."""
import json
import re
import sys
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tasks import make_sample  # noqa: E402


class _Tok:
    """make_sample needs a tokenizer only to size the haystack; reproduce the needles with the
    real tokenizer for exactness."""


def classify(out, answers, distractors):
    nums = re.findall(r"\d{5,}", out)
    if not nums:
        return "no_number"
    got = nums[0]
    if got in answers:
        return "correct_first"
    if got in distractors:
        return "distractor_value"
    best = max(sum(a == b for a, b in zip(got, ans)) for ans in answers)
    if best >= 4:
        return "near_miss_digits"
    return "other_number"


def main(path, methods, model="/home/kooki3/models/qwen2.5-1.5b-instruct"):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    recs = [json.loads(l) for l in open(path)]
    tabs = {m: Counter() for m in methods}
    for r in recs:
        smp = make_sample(r["task"], r["ctx"], tok, r["seed"])
        prompt = smp["prompt"]
        distract = []
        for cs, ce, tgt in smp["spans"]:
            if not tgt:
                distract += re.findall(r"\d{7}", prompt[cs:ce])
        for m in methods:
            res = r["res"].get(m)
            if res is None or res["score"] == 1.0:
                continue
            tabs[m][(r["task"], classify(res["out"], r["answers"], distract))] += 1
    for m, c in tabs.items():
        print(m)
        for (task, kind), v in sorted(c.items()):
            print(f"   {task:13s} {kind:18s} {v}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2].split(","))
