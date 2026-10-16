#!/usr/bin/env bash
# Direct-exit-code checks of src/mech_matrix.py v3 on the 4-prompt check file:
#   smoke + check list + manifest built from the check file -> 0
#   full mode with the check list -> 2; full mode with the 63-prompt list on the check file -> 2
#   full mode without --manifest -> 2; smoke with a manifest whose run identity was altered -> 2
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
I=/home/kooki3/docs/projects/attention-review/2026-10-16/hourly-mech-v2-1426/original-prompt-identities.json
T=/home/kooki3/.claude/jobs/a0424013/tmp
cd $WT
$PY src/make_mech_manifest.py runs/mech_check_16k.jsonl results/mech_prompts_check.json $I $T/mm_check_manifest.json > /dev/null
echo "manifest_rc=$?"
taskset -c 0-5 $PY src/mech_matrix.py runs/mech_check_16k.jsonl results/mech_prompts_check.json $I --manifest $T/mm_check_manifest.json --smoke > results/mech_matrix_smoke.log 2>&1
echo "smoke_rc=$?"
taskset -c 0-5 $PY src/mech_matrix.py runs/mech_check_16k.jsonl results/mech_prompts_check.json $I --manifest $T/mm_check_manifest.json > /dev/null 2>&1
echo "full_mode_with_check_list_rc=$?"
taskset -c 0-5 $PY src/mech_matrix.py runs/mech_check_16k.jsonl results/mech_prompts_full.json $I --manifest $T/mm_check_manifest.json > /dev/null 2>&1
echo "full_mode_with_full_list_on_check_file_rc=$?"
taskset -c 0-5 $PY src/mech_matrix.py runs/mech_check_16k.jsonl results/mech_prompts_check.json $I > /dev/null 2>&1
echo "full_mode_without_manifest_rc=$?"
$PY -c "import json,sys; m=json.load(open(sys.argv[1])); m['run']['defaults']['budget']=1024; json.dump(m,open(sys.argv[2],'w'))" $T/mm_check_manifest.json $T/mm_bad_manifest.json
taskset -c 0-5 $PY src/mech_matrix.py runs/mech_check_16k.jsonl results/mech_prompts_check.json $I --manifest $T/mm_bad_manifest.json --smoke > /dev/null 2>&1
echo "smoke_with_altered_manifest_rc=$?"
# N1-final-a: manifest with a wrong JSONL hash / wrong analysis-script hash must be refused
$PY -c "import json,sys; m=json.load(open(sys.argv[1])); m['mech_sha256']='0'*64; json.dump(m,open(sys.argv[2],'w'))" $T/mm_check_manifest.json $T/mm_bad_mech.json
taskset -c 0-5 $PY src/mech_matrix.py runs/mech_check_16k.jsonl results/mech_prompts_check.json $I --manifest $T/mm_bad_mech.json --smoke > /dev/null 2>&1
echo "smoke_with_wrong_mech_sha_rc=$?"
$PY -c "import json,sys; m=json.load(open(sys.argv[1])); m['mech_matrix_sha256']='0'*64; json.dump(m,open(sys.argv[2],'w'))" $T/mm_check_manifest.json $T/mm_bad_script.json
taskset -c 0-5 $PY src/mech_matrix.py runs/mech_check_16k.jsonl results/mech_prompts_check.json $I --manifest $T/mm_bad_script.json --smoke > /dev/null 2>&1
echo "smoke_with_wrong_script_sha_rc=$?"
# N1-final-b: an EXPECTED identity that does not match the rows must be refused (the frozen full-run EXPECTED file
# differs from the old check run's identity by construction: different argv/prompt list)
taskset -c 0-5 $PY src/mech_matrix.py runs/mech_check_16k.jsonl results/mech_prompts_check.json $I --manifest $T/mm_check_manifest.json --expected results/mech_full_expected.json --smoke > /dev/null 2>&1
echo "smoke_with_full_run_expected_rc=$?"
