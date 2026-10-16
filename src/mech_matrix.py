"""Theory v8 §4 descriptive evidence matrix on the mechanism diagnostic — v4 of this script (supersedes v1 9c5fb760…,
v2 f192fadb…, v3 bb254270…; after Codex reviews M1–M4, N1–N3 and N1-final-a/b). Definitions frozen
BEFORE the 63-prompt diagnostic finished.
No mechanism verdict; development data; not confirmatory.

GATE (full mode exits 2 on any failure):
  - PROMPTS must be exactly results/mech_prompts_full.json (63 keys) in full mode; --smoke accepts ONLY
    results/mech_prompts_check.json (4 keys) and prints "SMOKE — NO FINDINGS";
  - every row matches the frozen identity file (IDENTITIES): prompt_sha, L, answers; token_sha too when the frozen
    token_sha is not null; token_identity must be "verified" exactly when the frozen token_sha exists, else
    "first_recorded";
  - the existing acceptance (src/mech_check_accept.py MECH PROMPTS) must exit 0 on the same file;
  - EXPECTED identity (src/make_mech_expected.py, written from the launch script and frozen files BEFORE the run):
    required in full mode; every row's run must equal EXPECTED.run and the prompt list / identity file sha256 must
    equal EXPECTED's (an independent basis, not derived from the analysed output);
  - analysis manifest (MANIFEST, written by src/make_mech_manifest.py after the run and before any analysis): its
    mech_sha256 must equal the sha256 of the analysed JSONL and its mech_matrix_sha256 the sha256 of this script
    (missing fields fail); sha256 of the prompt list and identity file must match; every row's run identity (code, argv, model, model_id,
    identities_sha, defaults) must be identical across rows and equal to the manifest's; the identity file's sha256
    prefix must equal run.identities_sha. Required in full mode; in --smoke a manifest is optional;
  - state semantics: dense_tf.state_origin == "dense_TF", every arm tf == "arm_TF", fr == "arm_FR";
  - dense_weight arrays (used by E2) have length H, finite, non-negative values;
  - shape checks on what this script uses: dense_tf and every arm's tf have non-empty records and a state_origin;
    every candidate record has arrays of one common head count H for exact / full_moment / each scorer; in_topk values
    in {0, 0.5, 1}; selected values are booleans.
STATES: SHARED = dense_tf (exact block-LSE ranking, full second-order-moment ranking, and the three scorers on the
same hidden states); OWN = each arm's own teacher-forced trajectory (arms differ in state; reported separately).
RECORDS: one per (decode step j, layer >= 2); step j predicts cont[j]; digit index d = digit_of[j] (0..6, printed as
1..7). Only region == "candidate" is analysed; other regions are counted.
GROUPS: IN = unstraddled prompts; PRE / POST = straddled prompts whose target block is the head / tail probe block;
OTHER counted only.
ESTIMANDS (all per group, per scorer):
  (E1) prompt-equal share: within each prompt the share of (head, record) cells in a category, then the equal-weight
       mean over prompts having the group;
  (E2) dense-weight share: within each prompt sum(w * 1[category]) / sum(w) with w = dense_weight of the target block,
       then the equal-weight mean over prompts; prompts with zero total weight and prompts with missing weight are
       excluded and counted separately, and the total weight is reported.
CATEGORIES: exact ∈ {definite-in, definite-out, tie} × actual selected ∈ {yes, no} — a 3×2 table, mutually exclusive
and exhaustive over valid cells; marginals derived from it; "exact definite-in but not selected" is one cell.
UNCERTAINTY: the observed IN − POST difference of E1 for exact definite-in, full-moment definite-in, and each scorer's
selected share; 95% percentile interval from a prompt bootstrap (2000, seed 0, the same resample for all quantities);
a quantity with an empty group in a replicate is NA there; the number of valid replicates is printed for every
pre-declared quantity (0 if never valid).
STRATIFIED: task × digit (1..7) × group, E1 estimand, with n_prompts / n_records / n_cells.
OWN-TRAJECTORY TABLE: per arm, E1 shares of exact definite-in and own selected for IN / PRE / POST on the arm's own TF
state.
OUTCOME LINK (association only): per arm, straddled prompts are classified by their free run as no divergence within
the gold-continuation window (first_divergence is None: the generation starts with the whole gold continuation; extra
tokens such as a stop token may follow) / first divergence at a digit (and, among those, at the first POST digit:
digit index == number of value tokens in the head block, block size from the run's defaults) / at a non-digit token.
For digit divergences, the own-TF selected share of the target block at the divergent step and at the same prompt's
other digit steps is reported on the prompts where BOTH are available (paired), with the counts of prompts having only
one side and why; "other digit steps" are teacher-forced (gold prefix), a diagnostic state, not the free run's state.
Strict answer correctness (frozen score_strict on the free-run text) and gold_continuation_prefix_match are separate.
Usage: mech_matrix.py MECH.jsonl PROMPTS.json IDENTITIES.json [--manifest M.json] [--expected E.json] [--smoke]"""
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from tasks import score_strict  # noqa: E402

ARMS = ["cobs", "nf_pt1", "quest", "cobs+ftail"]
SCORERS = ["cobs", "nf_pt1", "quest"]
GROUPS = ["IN", "PRE", "POST"]
FULL = os.path.join(ROOT, "results/mech_prompts_full.json")
CHECK = os.path.join(ROOT, "results/mech_prompts_check.json")
EXACT_CATS = {1: "in", 0: "out", 0.5: "tie"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def gate(path, rows, prompts_path, idents, smoke, idents_path=None, manifest=None, expected=None):
    bad = []
    if manifest is None and not smoke:
        bad.append("full mode needs --manifest (src/make_mech_manifest.py)")
    if expected is None and not smoke:
        bad.append("full mode needs --expected (src/make_mech_expected.py)")
    if expected is not None:
        if expected.get("prompts_sha256") != sha(prompts_path) or expected.get("identities_sha256") != sha(idents_path):
            bad.append("prompt list / identity file differ from the EXPECTED identity")
        if any(r.get("run") != expected.get("run") for r in rows):
            bad.append("row run identity differs from the EXPECTED identity")
    if manifest is not None:
        if not manifest.get("mech_sha256") or manifest["mech_sha256"] != sha(path):
            bad.append("manifest mech_sha256 missing or differs from the analysed JSONL")
        if not manifest.get("mech_matrix_sha256") or manifest["mech_matrix_sha256"] != sha(os.path.abspath(__file__)):
            bad.append("manifest mech_matrix_sha256 missing or differs from this analysis script")
        if manifest.get("prompts_sha256") != sha(prompts_path) or manifest.get("identities_sha256") != sha(idents_path):
            bad.append("prompt list / identity file differ from the analysis manifest")
    runs = {json.dumps(r.get("run"), sort_keys=True) for r in rows}
    if len(runs) != 1:
        bad.append(f"{len(runs)} distinct run identities in one file")
    elif manifest is not None and json.loads(next(iter(runs))) != manifest.get("run"):
        bad.append("run identity differs from the analysis manifest")
    if rows and idents_path and rows[0].get("run", {}).get("identities_sha") != sha(idents_path)[:16]:
        bad.append("run.identities_sha differs from the identity file")
    want = FULL if not smoke else CHECK
    if os.path.abspath(prompts_path) != want:
        bad.append(f"prompt list must be {os.path.relpath(want, ROOT)} in this mode")
    prompts = [tuple(p) for p in json.load(open(prompts_path))]
    keys = [(r["task"], r["seed"]) for r in rows]
    if sorted(keys) != sorted(prompts) or len(set(keys)) != len(keys):
        bad.append(f"row keys differ from the prompt list ({len(keys)} rows / {len(prompts)} prompts)")
    for r in rows:
        tag = f"{r['task']}:{r['seed']}"
        fz = idents.get((r["task"], r["seed"]))
        if fz is None:
            bad.append(f"{tag}: no frozen identity")
            continue
        if r.get("prompt_sha") != fz["prompt_sha"] or r.get("L") != fz["L"] or \
                (fz.get("answers") is not None and [r.get("value")] != fz["answers"]):
            bad.append(f"{tag}: prompt/L/answer differ from the frozen identity")
        want_ti = "verified" if fz.get("token_sha") is not None else "first_recorded"
        ti = r.get("token_identity")
        if ti is None and smoke and want_ti == "verified":      # the 4-prompt check file predates the field
            ti = "verified"
        if ti != want_ti or (want_ti == "verified" and r.get("token_sha") != fz["token_sha"]):
            bad.append(f"{tag}: token identity {r.get('token_identity')} (want {want_ti})")
        if set(r.get("arms", {})) != set(ARMS):
            bad.append(f"{tag}: arms {sorted(r.get('arms', {}))}")
            continue
        states = [("dense_tf", r.get("dense_tf"), "dense_TF")] + \
                 [(f"{a}.tf", r["arms"][a].get("tf"), "arm_TF") for a in ARMS]
        for a in ARMS:
            if (r["arms"][a].get("fr") or {}).get("state_origin") != "arm_FR":
                bad.append(f"{tag}: {a}.fr state_origin != arm_FR")
        for name, st, origin in states:
            if not st or not st.get("records") or st.get("state_origin") != origin:
                bad.append(f"{tag}: {name} empty or state_origin != {origin}")
                continue
            for x in st["records"]:
                if x.get("region") != "candidate":
                    continue
                H = len(x["exact"]["in_topk"])
                cats = [x["exact"]["in_topk"], x["full_moment"]["in_topk"]] + [v["in_topk"] for v in x["scorers"].values()]
                sels = [v["selected"] for v in x["scorers"].values()]
                w = x.get("dense_weight")
                w_bad = w is not None and (len(w) != H or any(not (isinstance(v, (int, float)) and math.isfinite(v)
                                                                  and v >= 0) for v in w))
                if H <= 0 or w_bad or any(len(a) != H for a in cats + sels) or \
                        any(v not in (0, 0.5, 1) for a in cats for v in a) or \
                        any(not isinstance(v, bool) for a in sels for v in a):
                    bad.append(f"{tag}: {name} record step {x.get('step')} layer {x.get('layer')} malformed")
                    break
    acc = subprocess.run([sys.executable, os.path.join(HERE, "mech_check_accept.py"), path, prompts_path],
                         capture_output=True, text=True)
    if acc.returncode != 0:
        bad.append(f"mech_check_accept exit {acc.returncode}: {acc.stdout.strip().splitlines()[-1:]}")
    return bad


def group_of(r, x):
    if not r["straddle"]:
        return "IN"
    pb = x["probe_blocks"]
    return "PRE" if x["target_block"] == pb["head"] else "POST" if x["target_block"] == pb["tail"] else "OTHER"


def cells_of(r, records, scorer):
    """Group -> list of cells; region counts."""
    out, regions = defaultdict(list), defaultdict(int)
    for x in records:
        regions[x.get("region", "?")] += 1
        if x.get("region") != "candidate":
            continue
        g = group_of(r, x)
        sc = x["scorers"][scorer] if scorer in x["scorers"] else x["scorers"][scorer.split("+")[0]]
        d = r["digit_of"][x["step"]] if 0 <= x["step"] < len(r["digit_of"]) else None
        w = x.get("dense_weight")
        for h in range(len(x["exact"]["in_topk"])):
            out[g].append(dict(exact=EXACT_CATS[x["exact"]["in_topk"][h]], moment=EXACT_CATS[x["full_moment"]["in_topk"][h]],
                               sel=sc["selected"][h], w=None if w is None else w[h], step=x["step"],
                               digit=d, task=r["task"], rec=(x["step"], x["layer"])))
    return out, regions


WEIGHT_EXCLUDED = defaultdict(int)


def table(cells, weighted=False):
    """3x2 table (exact category x selected) as shares; None if no (weighted) mass."""
    if not cells:
        return None
    keys = [(e, s) for e in ("in", "out", "tie") for s in (True, False)]
    if weighted:
        ws = [c["w"] for c in cells]
        if any(w is None for w in ws):
            WEIGHT_EXCLUDED["missing"] += 1
            return None
        if sum(ws) <= 0:
            WEIGHT_EXCLUDED["zero_mass"] += 1
            return None
        WEIGHT_EXCLUDED["total_weight"] += sum(ws)
        tot = sum(ws)
        return {f"{e}/{'sel' if s else 'not'}": sum(c["w"] for c in cells if c["exact"] == e and c["sel"] == s) / tot
                for e, s in keys}
    n = len(cells)
    return {f"{e}/{'sel' if s else 'not'}": sum(1 for c in cells if c["exact"] == e and c["sel"] == s) / n
            for e, s in keys}


def prompt_mean(per_prompt, g, fn):
    vals = [fn(p[g]) for p in per_prompt if p[g]]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0
    return {k: sum(v[k] for v in vals) / len(vals) for k in vals[0]}, len(vals)


def e1(cells, what):
    if not cells:
        return None
    if what == "exact_in":
        return sum(c["exact"] == "in" for c in cells) / len(cells)
    if what == "moment_in":
        return sum(c["moment"] == "in" for c in cells) / len(cells)
    return sum(c["sel"] for c in cells) / len(cells)


def in_minus_post(per, idx, what):
    a = [e1(per[i]["IN"], what) for i in idx if per[i]["IN"]]
    b = [e1(per[i]["POST"], what) for i in idx if per[i]["POST"]]
    return None if not a or not b else sum(a) / len(a) - sum(b) / len(b)


def fmt(d):
    return "NA" if d is None else "{" + ", ".join(f"{k} {v:.3f}" for k, v in d.items()) + "}"


def main(path, prompts_path, idents_path, smoke, manifest_path=None, expected_path=None):
    rows = [json.loads(l) for l in open(path)]
    idents = {(x["task"], x["seed"]): x for x in json.load(open(idents_path))}
    manifest = json.load(open(manifest_path)) if manifest_path else None
    expected = json.load(open(expected_path)) if expected_path else None
    bad = gate(path, rows, prompts_path, idents, smoke, idents_path, manifest, expected)
    if bad:
        print("GATE FAILED:", *bad[:15], sep="\n  ")
        sys.exit(2)
    if smoke:
        print("SMOKE — NO FINDINGS (implementation check only)")
    print(f"prompts {len(rows)}; straddled {sum(r['straddle'] for r in rows)}")
    shared, regions = {}, defaultdict(int)
    for s in SCORERS:
        shared[s] = []
        for r in rows:
            c, reg = cells_of(r, r["dense_tf"]["records"], s)
            shared[s].append(c)
            if s == SCORERS[0]:
                for k, v in reg.items():
                    regions[k] += v
    print("records by region (shared state):", dict(regions),
          "| OTHER-group cells:", sum(len(p["OTHER"]) for p in shared[SCORERS[0]]))

    print("\n== SHARED state: 3x2 (exact category x selected), E1 prompt-equal and E2 dense-weight ==")
    for g in GROUPS:
        n_rec = sum(len({c["rec"] for c in p[g]}) for p in shared[SCORERS[0]])
        n_cells = sum(len(p[g]) for p in shared[SCORERS[0]])
        for s in SCORERS:
            t1, n1 = prompt_mean(shared[s], g, table)
            WEIGHT_EXCLUDED.clear()
            t2, n2 = prompt_mean(shared[s], g, lambda c: table(c, True))
            print(f"{g:4s} {s:7s} prompts {n1} records {n_rec} cells {n_cells}\n     E1 {fmt(t1)}\n     E2 {fmt(t2)} "
                  f"(prompts used {n2}; excluded zero-mass {WEIGHT_EXCLUDED['zero_mass']}, missing "
                  f"{WEIGHT_EXCLUDED['missing']}; total weight {WEIGHT_EXCLUDED['total_weight']:.4g})")
        mo, _ = prompt_mean(shared[SCORERS[0]], g, lambda c: {"moment_in": e1(c, "moment_in")} if c else None)
        print(f"     full-moment definite-in (E1) {fmt(mo)}")

    print("\n== IN − POST (E1), observed difference and prompt-bootstrap 95% interval ==")
    rng = random.Random(0)
    idx_all = list(range(len(rows)))
    quantities = [("exact_in", SCORERS[0]), ("moment_in", SCORERS[0])] + [("sel", s) for s in SCORERS]
    obs = {(q, s): in_minus_post(shared[s], idx_all, q) for q, s in quantities}
    boots = {(q, s): [] for q, s in quantities}
    for _ in range(2000):
        smp = [rng.choice(idx_all) for _ in idx_all]
        for q, s in quantities:
            d = in_minus_post(shared[s], smp, q)
            if d is not None:
                boots[(q, s)].append(d)
    for (q, s), v in boots.items():
        v = sorted(v)
        ci = f"[{v[int(.025 * len(v))]:+.3f}, {v[int(.975 * len(v)) - 1]:+.3f}]" if v else "NA"
        o = "NA" if obs[(q, s)] is None else f"{obs[(q, s)]:+.3f}"
        name = {"exact_in": "exact definite-in", "moment_in": "full-moment definite-in"}.get(q, f"selected:{s}")
        print(f"  {name:24s} observed {o}  95% {ci}  valid replicates {len(v)}")

    print("\n== STRATIFIED (E1 selected share, shared state): group / task / digit (1..7) ==")
    for s in SCORERS:
        strata = defaultdict(list)
        for p in shared[s]:
            for g in GROUPS:
                by = defaultdict(list)
                for c in p[g]:
                    by[(c["task"], c["digit"])].append(c)
                for k, cs in by.items():
                    strata[(g,) + k].append(cs)
        parts = []
        for (g, t, d), lst in sorted(strata.items(), key=lambda kv: str(kv[0])):
            m = sum(e1(cs, "sel") for cs in lst) / len(lst)
            parts.append(f"{g}/{t}/d{'?' if d is None else d + 1}={m:.2f}"
                         f"(p={len(lst)},r={sum(len({c['rec'] for c in cs}) for cs in lst)},c={sum(len(cs) for cs in lst)})")
        print(f"  {s}: " + "; ".join(parts))

    print("\n== OWN trajectories (each arm's own TF state; states differ between arms) ==")
    for arm in ARMS:
        s = arm                                   # own-arm scorer key (e.g. "cobs+ftail" records its forced mask)
        per = [cells_of(r, r["arms"][arm]["tf"]["records"], s)[0] for r in rows]
        for g in GROUPS:
            ex, n = prompt_mean(per, g, lambda c: {"exact_in": e1(c, "exact_in"), "selected": e1(c, "sel")}
                                if c else None)
            print(f"  {arm:10s} {g:4s} prompts {n} {fmt(ex)}")

    print("\n== OUTCOME LINK (association only), straddled prompts ==")
    for arm in ARMS:
        s = arm
        cnt = {k: 0 for k in ("all", "no_divergence_in_gold_window", "digit_divergence", "at_first_post_digit",
                              "non_digit_divergence", "strict_correct", "gold_continuation_prefix_match",
                              "link_paired", "link_only_divergent_step", "link_only_other_steps", "link_neither")}
        at_div, other = [], []
        B = rows[0]["run"]["defaults"]["block"]
        for r in (r for r in rows if r["straddle"]):
            fr = r["arms"][arm]["fr"]
            cnt["all"] += 1
            cnt["strict_correct"] += int(score_strict(r["task"], fr.get("text", ""), [r["value"]]))
            fd = fr.get("first_divergence")
            cnt["gold_continuation_prefix_match"] += int(fd is None)
            if fd is None:
                cnt["no_divergence_in_gold_window"] += 1
                continue
            d = r["digit_of"][fd] if fd < len(r["digit_of"]) else None
            if d is None:
                cnt["non_digit_divergence"] += 1
                continue
            cnt["digit_divergence"] += 1
            head_digits = sum(1 for t in r["src_tokens"] if t // B == r["src_tokens"][0] // B)
            cnt["at_first_post_digit"] += int(d == head_digits)
            cells, _ = cells_of(r, r["arms"][arm]["tf"]["records"], s)
            allc = [c for g in GROUPS for c in cells[g]]
            a = [c for c in allc if c["step"] == fd]
            o = [c for c in allc if c["step"] != fd and c["digit"] is not None]
            if a and o:
                cnt["link_paired"] += 1
                at_div.append(e1(a, "sel"))
                other.append(e1(o, "sel"))
            else:
                cnt["link_only_divergent_step" if a else "link_only_other_steps" if o else "link_neither"] += 1

        def m(v):
            return "NA (prompts 0)" if not v else f"{sum(v) / len(v):.3f} (prompts {len(v)})"
        print(f"  {arm:10s} {cnt}\n             paired prompts: own-TF selected share at divergent step {m(at_div)}; "
              f"at other (teacher-forced) digit steps {m(other)}")
    if smoke:
        print("\nSMOKE — NO FINDINGS")


if __name__ == "__main__":
    extra = sys.argv[4:]
    mp = extra[extra.index("--manifest") + 1] if "--manifest" in extra else None
    ep = extra[extra.index("--expected") + 1] if "--expected" in extra else None
    main(sys.argv[1], sys.argv[2], sys.argv[3], "--smoke" in extra, mp, ep)
