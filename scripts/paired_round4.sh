#!/usr/bin/env bash
# Round-4 paired comparisons (same seeds across runs).
cd "$(dirname "$0")/../src"
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
for b in 512 256; do
  echo "== budget $b"
  $PY compare_runs.py questB16=../runs/quest_b16_16k.jsonl:quest@$b nf_hi1=../runs/nf_16k.jsonl:nf_hi1@$b \
    nf_hi2=../runs/nf_16k.jsonl:nf_hi2@$b nf_dual2=../runs/nf_16k.jsonl:nf_dual2@$b \
    sum_dual=../runs/nf_16k.jsonl:sum_dual@$b questB32=../runs/nf_16k.jsonl:quest@$b
  echo
  $PY compare_runs.py questB32=../runs/nf_16k.jsonl:quest@$b nf_hi1=../runs/nf_16k.jsonl:nf_hi1@$b \
    nf_hi2=../runs/nf_16k.jsonl:nf_hi2@$b
  echo
done
