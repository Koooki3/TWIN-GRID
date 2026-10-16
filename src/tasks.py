"""RULER-style needle tasks, generated locally (not the official RULER harness).

Templates follow RULER's niah_* tasks: needles are "One of the special magic
numbers for {key} is: {value}." and the answer prefix is placed in the
assistant turn so the model answers immediately.
"""
import glob
import random
import re

ESSAY_DIR = "/home/kooki3/datasets/gutenberg-essays"
NOISE = ("The grass is green. The sky is blue. The sun is yellow. "
         "Here we go. There and back again. ")

WORDS = ("apple river candle marble violet harbor falcon copper meadow lantern "
         "orchid thunder velvet saddle quartz pepper glacier ember walnut cobalt "
         "jasmine anchor timber crystal ferret summit willow beacon raven tulip").split()


def _load_essays():
    out = []
    for p in sorted(glob.glob(f"{ESSAY_DIR}/pg*.txt")):
        t = open(p, encoding="utf-8").read()
        s = t.find("*** START")
        e = t.find("*** END")
        t = t[t.find("\n", s) + 1:e] if s >= 0 and e > s else t
        out.append(re.sub(r"\s+", " ", t))
    return " ".join(out)


_ESSAYS = None
CORPUS_V2_DIR = "/home/kooki3/datasets/gutenberg-nonfiction"   # 47 public-domain books, MANIFEST.sha256
SLOT_CHARS_PER_TOKEN = 6        # v2 slot length = ctx_tokens * 6 chars (calibrated haystacks need ~4.2-4.6)
_V2 = None


_V2_BOOKS = []                   # [(file name, start offset in the cleaned corpus string)]


def _load_v2():
    """Load the cleaned v2 corpus. Every pg*.txt must match MANIFEST.sha256 exactly (same file set and hashes)."""
    import hashlib
    man = {}
    for line in open(f"{CORPUS_V2_DIR}/MANIFEST.sha256"):
        h, name = line.split()
        man[name] = h
    files = sorted(glob.glob(f"{CORPUS_V2_DIR}/pg*.txt"))
    names = [p.rsplit("/", 1)[1] for p in files]
    if sorted(names) != sorted(man):
        raise RuntimeError(f"v2 corpus file set differs from MANIFEST.sha256: {sorted(set(names) ^ set(man))}")
    out, pos = [], 0
    _V2_BOOKS.clear()
    for p, name in zip(files, names):
        raw = open(p, "rb").read()
        if hashlib.sha256(raw).hexdigest() != man[name]:
            raise RuntimeError(f"v2 corpus file {name} does not match MANIFEST.sha256")
        t = raw.decode("utf-8-sig")
        s = t.find("*** START")
        e = t.find("*** END")
        t = t[t.find("\n", s) + 1:e] if s >= 0 and e > s else t
        t = re.sub(r"\s+", " ", t)
        _V2_BOOKS.append((name, pos))
        out.append(t)
        pos += len(t) + 1
    return " ".join(out)


def v2_books(start, end):
    """Source books (file names) covering corpus characters [start, end) — provenance recorded per prompt."""
    return [name for i, (name, b0) in enumerate(_V2_BOOKS)
            if b0 < end and (_V2_BOOKS[i + 1][1] if i + 1 < len(_V2_BOOKS) else float("inf")) > start]


def v2_slots(ctx_tokens):
    """Number of disjoint haystack windows available at this context length."""
    global _V2
    if _V2 is None:
        _V2 = _load_v2()
    return len(_V2) // (ctx_tokens * SLOT_CHARS_PER_TOKEN)


def _v2_window(slot, ctx_tokens, n_chars):
    global _V2
    if _V2 is None:
        _V2 = _load_v2()
    L = ctx_tokens * SLOT_CHARS_PER_TOKEN
    if not 0 <= slot < len(_V2) // L:
        raise ValueError(f"slot {slot} out of range ({len(_V2) // L} disjoint slots at ctx {ctx_tokens})")
    if n_chars > L:
        raise ValueError(f"haystack needs {n_chars} chars > slot length {L}")
    start = slot * L
    return start, _V2[start:start + n_chars]


def haystack_sentences(kind, n_chars, rng, slot=None, ctx_tokens=None):
    global _ESSAYS
    if slot is not None:                       # v2: disjoint window per prompt (independent haystacks)
        _, text = _v2_window(slot, ctx_tokens, n_chars)
        return re.findall(r"[^.!?]+[.!?]\s*", text)
    if kind == "noise":
        reps = n_chars // len(NOISE) + 1
        text = NOISE * reps
        return re.findall(r"[^.]+\.\s*", text)
    if _ESSAYS is None:
        _ESSAYS = _load_essays()
    start = rng.randrange(0, max(1, len(_ESSAYS) - n_chars - 1))
    text = _ESSAYS[start:start + n_chars]
    return re.findall(r"[^.!?]+[.!?]\s*", text)


def _num(rng):
    return str(rng.randrange(1_000_000, 10_000_000))


def make_sample(task, ctx_tokens, tok, seed, slot=None):
    """Return dict(prompt, answers, needle_spans(char), key). Haystack is sized
    so the full chat prompt is close to ctx_tokens. slot=None: v1 haystack (random offset into the 3-book corpus,
    windows of different prompts may overlap); slot=int: v2 haystack, the slot-th disjoint window of the 47-book
    corpus, so distinct slots share no haystack text."""
    rng = random.Random(seed)
    hay = "noise" if task == "single_noise" else "essay"
    key = rng.choice(WORDS)
    needles = []  # (text, is_target)
    answers = []
    if task in ("single_noise", "single_essay"):
        v = _num(rng)
        needles.append((f"One of the special magic numbers for {key} is: {v}.", True))
        answers = [v]
    elif task == "multikey":
        others = rng.sample([w for w in WORDS if w != key], 3)
        v = _num(rng)
        needles.append((f"One of the special magic numbers for {key} is: {v}.", True))
        for o in others:
            needles.append((f"One of the special magic numbers for {o} is: {_num(rng)}.", False))
        answers = [v]
    elif task == "multivalue":
        for _ in range(4):
            v = _num(rng)
            needles.append((f"One of the special magic numbers for {key} is: {v}.", True))
            answers.append(v)
    else:
        raise ValueError(task)

    if task == "multivalue":
        q = (f"What are all the special magic numbers for {key} mentioned in the provided text?")
        prefix = f"The special magic numbers for {key} mentioned in the provided text are"
    else:
        q = f"What is the special magic number for {key} mentioned in the provided text?"
        prefix = f"The special magic number for {key} mentioned in the provided text is"

    head = ("Some special magic numbers are hidden within the following text. "
            "Make sure to memorize it. I will quiz you about the numbers afterwards.\n")
    tail = f"\n{q}"

    def build(n_chars):
        sents = haystack_sentences(hay, n_chars, random.Random(seed + 1), slot, ctx_tokens)
        pos = sorted(rng2.randrange(0, len(sents) + 1) for _ in needles)
        order = list(needles)
        random.Random(seed + 2).shuffle(order)
        body, spans, j = "", [], 0
        for i in range(len(sents) + 1):
            while j < len(pos) and pos[j] == i:
                body += " " + order[j][0] + " "
                spans.append((len(head) + len(body) - len(order[j][0]) - 1,
                              len(head) + len(body) - 1, order[j][1]))
                j += 1
            if i < len(sents):
                body += sents[i]
        user = head + body + tail
        msgs = [{"role": "user", "content": user}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        off = chat.find(user)
        spans = [(a + off, b + off, t) for a, b, t in spans]
        return chat + prefix, spans

    rng2 = random.Random(seed + 3)
    n_chars = ctx_tokens * 4
    for _ in range(4):  # calibrate chars/token
        rng2 = random.Random(seed + 3)
        prompt, spans = build(n_chars)
        n = len(tok(prompt).input_ids)
        n_chars = int(n_chars * ctx_tokens / n)
        if abs(n - ctx_tokens) < 0.02 * ctx_tokens:
            break
    rng2 = random.Random(seed + 3)
    prompt, spans = build(n_chars)
    st0 = None if slot is None else slot * ctx_tokens * SLOT_CHARS_PER_TOKEN
    hay_info = None if slot is None else dict(corpus="v2", slot=slot, start=st0, n_chars=n_chars,
                                              books=v2_books(st0, st0 + n_chars))
    return dict(prompt=prompt, answers=answers, spans=spans, key=key, task=task, hay=hay_info)


def score(task, text, answers):
    """Answer containment (substring) — the metric used in exploration runs."""
    if task == "multivalue":
        return sum(a in text for a in answers) / len(answers)
    return float(answers[0] in text)


def score_strict(task, text, answers):
    """Strict match: a target value counts only as a maximal digit run of the output (digit boundaries).
    Multivalue: fraction of target values found this way (exact-set match is all of them)."""
    runs = set(re.findall(r"\d+", text))
    if task == "multivalue":
        return sum(a in runs for a in answers) / len(answers)
    return float(answers[0] in runs)
