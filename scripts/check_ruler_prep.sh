#!/usr/bin/env bash
# Direct-exit-code checks of src/ruler_prep_manifest.py on copies of the sealed 16k smoke data (no tokenizer, no
# regeneration): unmanifested old data -> begin refused (1); sealed file drifted (same line count) -> begin refused (1)
# and finish refused (1); untouched sealed copy -> begin ok (0) and finish ok (0).
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
SRC=$WT/runs/ruler/qwen1.5b_16k_smoke2
T=/home/kooki3/.claude/jobs/a0424013/tmp/prepcheck
rm -rf $T; mkdir -p $T
# a) old data without manifest
mkdir -p $T/a; cp -r $SRC/niah_single_2 $SRC/qa_1 $T/a/
$PY $WT/src/ruler_prep_manifest.py begin $T/a 8192 5 niah_single_2 qa_1 > /dev/null; echo "unmanifested_old_data_begin_rc=$?"
# b) sealed copy, untouched — the sealed manifest predates the tokenizer-file field change, so reseal a copy first
mkdir -p $T/b; cp -r $SRC/niah_single_2 $SRC/qa_1 $T/b/
$PY - $T/b <<'EOF'
import json, sys, os
sys.path.insert(0, "/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis/src")
import ruler_prep_manifest as m
cfg = m.config(16384, 5, ["niah_single_2", "qa_1"])
json.dump(cfg, open(os.path.join(sys.argv[1], "prep_manifest.json"), "w"), indent=1)
EOF
$PY $WT/src/ruler_prep_manifest.py finish $T/b > /dev/null; echo "seal_copy_finish_rc=$?"
$PY $WT/src/ruler_prep_manifest.py begin $T/b 16384 5 niah_single_2 qa_1 > /dev/null; echo "untouched_sealed_begin_rc=$?"
$PY $WT/src/ruler_prep_manifest.py finish $T/b > /dev/null; echo "untouched_sealed_finish_rc=$?"
# c) drift: change one character in a sealed file, same line count
sed -i '1s/a/b/' $T/b/qa_1/validation.jsonl
$PY $WT/src/ruler_prep_manifest.py begin $T/b 16384 5 niah_single_2 qa_1 > /dev/null; echo "drifted_sealed_begin_rc=$?"
$PY $WT/src/ruler_prep_manifest.py finish $T/b > /dev/null; echo "drifted_sealed_finish_rc=$?"
rm -rf $T
