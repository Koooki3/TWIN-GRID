"""Before resuming after an interruption (reviewer, plan A for the RECAP GPU window): ruler_pred writes one line + flush
per sample and json.loads every existing line on resume, so a signal landing mid-write could leave a truncated last line
and make the resume fail. Two passes over every prediction .jsonl under the given root:
  pass 1 (read only): parse every line, check for duplicate sample_uids; collect problems and repair candidates (files
          whose ONLY problem is an incomplete last line);
  pass 2 (only with --fix and only if pass 1 found no other problem anywhere): archive each candidate file unchanged
          to <root>/_interrupted/<timestamp>/ and remove just its incomplete last line (an empty file if that was the
          only line). Any other problem -> exit 2 and nothing is modified (reviewer: two defects fixed, self-test below).
Usage: check_jsonl_tail.py PRED_ROOT [--fix]     |     check_jsonl_tail.py --selftest"""
import json
import os
import shutil
import sys
import tempfile
import time


def scan(root):
    bad, cand = [], []
    for dp, _, fs in os.walk(root):
        if "_interrupted" in dp:
            continue
        for f in fs:
            if not f.endswith(".jsonl"):
                continue
            p = os.path.join(dp, f)
            lines = open(p, "rb").read().split(b"\n")
            if lines and lines[-1] == b"":
                lines = lines[:-1]
            uids, broken = [], []
            for k, ln in enumerate(lines):
                try:
                    uids.append(json.loads(ln)["sample_uid"])
                except Exception:
                    broken.append(k)
            if len(uids) != len(set(uids)):
                bad.append(f"{p}: duplicate uids")
            if broken and broken != [len(lines) - 1]:
                bad.append(f"{p}: unparsable lines {broken} (not only the last one)")
            elif broken:
                cand.append((p, lines))
    return bad, cand


def repair(root, cand):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = []
    for p, lines in cand:
        arch = os.path.join(root, "_interrupted", stamp, os.path.relpath(p, root))
        os.makedirs(os.path.dirname(arch), exist_ok=True)
        shutil.copy2(p, arch)
        keep = lines[:-1]
        open(p, "wb").write(b"\n".join(keep) + b"\n" if keep else b"")
        out.append(f"{p}: incomplete last line removed (archived {arch})")
    return out


def main(root, fix):
    bad, cand = scan(root)
    if bad:
        print("PROBLEMS (nothing modified):", *bad, sep="\n  ")
        if cand:
            print("  also incomplete last lines (not repaired because of the problems above):", *[c[0] for c in cand],
                  sep="\n    ")
        return 2
    if cand and not fix:
        print("incomplete last lines (run with --fix to archive and remove them):", *[c[0] for c in cand], sep="\n  ")
        return 2
    for s in repair(root, cand):
        print(s)
    print(f"all prediction files under {root} parse; no duplicate uids" + (f"; repaired {len(cand)}" if cand else ""))
    return 0


def selftest():
    ok = True
    good = '{"sample_uid": "x:0", "pred": "a"}\n'
    d = tempfile.mkdtemp()
    open(os.path.join(d, "one.jsonl"), "w").write('{"sample_uid": "x:0"')          # single truncated line
    rc = main(d, True)
    after = open(os.path.join(d, "one.jsonl"), "rb").read()
    ok &= rc == 0 and after == b"" and main(d, False) == 0
    print(f"selftest single truncated line -> empty file: {after == b''}, rc {rc}")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "dup.jsonl")
    open(p, "w").write(good + good + '{"sample_uid": "x:1"')                         # duplicate + truncated tail
    before = open(p, "rb").read()
    rc = main(d, True)
    ok &= rc == 2 and open(p, "rb").read() == before and not os.path.exists(os.path.join(d, "_interrupted"))
    print(f"selftest duplicate + truncated: rc {rc}, file untouched {open(p, 'rb').read() == before}")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "tail.jsonl")
    open(p, "w").write(good + '{"sample_uid": "x:1"')                                # normal repair
    rc = main(d, True)
    ok &= rc == 0 and open(p).read() == good
    print(f"selftest normal truncated tail repaired: {open(p).read() == good}, rc {rc}")
    print("SELFTEST OK" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    sys.exit(main(sys.argv[1], "--fix" in sys.argv))
