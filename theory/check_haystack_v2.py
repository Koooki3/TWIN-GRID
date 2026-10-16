"""v2 haystack checks (tokenizer only, no model; the runner's local tokenizer): corpus matches MANIFEST.sha256 (checked
by the loader), slot counts, calibrated length fits the slot, windows of different slots are disjoint, provenance
(source books) recorded, deterministic regeneration, and ALL 80 phase-1 v1 prompts reproduce their recorded prompt and
token hashes. Exit 1 on failure."""
import sys

sys.path.insert(0, __file__.rsplit("/theory/", 1)[0] + "/src")
import tasks  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL = "/home/kooki3/models/qwen2.5-1.5b-instruct"          # same path as run_e2e.py / run_digit_diag.py
tok = AutoTokenizer.from_pretrained(MODEL)
bad = []
for ctx in (16384, 32768):
    print("ctx", ctx, "disjoint slots", tasks.v2_slots(ctx))
spans = []
for slot, task in ((0, "multikey"), (1, "single_essay"), (2, "multikey"), (tasks.v2_slots(16384) - 1, "single_essay")):
    smp = tasks.make_sample(task, 16384, tok, 210000 + slot, slot)
    n = len(tok(smp["prompt"]).input_ids)
    h = smp["hay"]
    L = 16384 * tasks.SLOT_CHARS_PER_TOKEN
    print(task, slot, "tokens", n, "chars", h["n_chars"], "slot len", L, "books", h["books"])
    if not h["books"]:
        bad.append(f"slot {slot}: no provenance")
    if not (h["n_chars"] <= L and abs(n - 16384) < 0.02 * 16384):
        bad.append(f"slot {slot}: length")
    spans.append((h["start"], h["start"] + h["n_chars"]))
    again = tasks.make_sample(task, 16384, tok, 210000 + slot, slot)
    if again["prompt"] != smp["prompt"]:
        bad.append(f"slot {slot}: not deterministic")
spans.sort()
if any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1)):
    bad.append("overlapping windows")
import hashlib, json  # noqa: E401,E402
import numpy as np  # noqa: E402
ref = {(r["task"], r["seed"]): (r["prompt_sha"], r["token_sha"]) for r in map(json.loads, open(
    __file__.rsplit("/theory/", 1)[0] + "/runs/phase1_g2f_dev.jsonl"))}
for (t, sd), (psha, tsha) in sorted(ref.items()):
    v1 = tasks.make_sample(t, 16384, tok, sd)
    ids = tok(v1["prompt"], return_tensors="np").input_ids.astype(np.int64)
    got = (hashlib.sha256(v1["prompt"].encode()).hexdigest()[:16], hashlib.sha256(ids.tobytes()).hexdigest()[:16])
    if v1["hay"] is not None or got != (psha, tsha):
        bad.append(f"v1 prompt changed {t} {sd}")
print("v1 prompts checked", len(ref))
print("PROBLEMS", bad if bad else "none")
sys.exit(1 if bad else 0)
