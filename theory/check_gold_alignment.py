"""CPU check (tokenizer only): the gold continuation extends the prompt at token level, and every digit maps to
one continuation step. Exit 1 on failure."""
import sys

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "src")
from transformers import AutoTokenizer  # noqa: E402

import run_digit_diag as R  # noqa: E402
from tasks import make_sample  # noqa: E402

tok = AutoTokenizer.from_pretrained("/home/kooki3/models/qwen2.5-1.5b-instruct")
ok = True
for task, seed in (("multikey", 1003), ("single_essay", 1018), ("multikey", 1000), ("single_essay", 1001)):
    smp = make_sample(task, 2048, tok, seed)
    ids_p, cont, dig = R.gold_continuation(tok, smp["prompt"], smp["answers"][0])
    digits = [d for d in dig if d is not None]
    good = digits == list(range(len(smp["answers"][0])))
    ok &= good
    print(task, seed, smp["answers"][0], [tok.decode([c]) for c in cont], dig, "OK" if good else "BAD")
print("ALL OK" if ok else "PROBLEM")
sys.exit(0 if ok else 1)
