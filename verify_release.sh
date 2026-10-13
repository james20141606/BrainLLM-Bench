#!/bin/bash
# Offline checks for this release (Python 3.9+ with numpy, scipy, pandas). Exits nonzero on any failure.
set -u
cd "$(dirname "$0")"
PY=${PY:-python3}
fails=0
run () {  # $1 expected substring, rest: command
  local want=$1; shift
  out=$("$@" 2>&1); rc=$?
  last=$(printf '%s\n' "$out" | tail -1)
  if [ $rc -eq 0 ] && printf '%s' "$out" | grep -qF "$want"; then echo "PASS  $last"
  else echo "FAIL  ($*) rc=$rc: $last"; fails=$((fails+1)); fi
}
run "198/198 claims reproduce" $PY -m bench.reproduce
run "Table 1 receipt pin: PASS" $PY codes/check_table1_receipts.py
run "PASS: 174/174 pins hold" $PY codes/check_paperA_main_numbers.py
run "PASS: paperA R1 pins" $PY codes/check_paperA_revision_r1_20261007.py
run "NEW follow-up receipt pin PASS" $PY codes/check_followup_receipts_20260923.py
run "C4 receipt pin: PASS" $PY codes/check_c4_receipt_20260924.py
run "Space-fix pin: PASS" $PY codes/check_space_fix_20260926.py
run "7/7 mutation cases behaved as expected" $PY codes/check_bench_space_fix_mutations_20260925.py
run "Extraction provenance pin: PASS" $PY codes/check_extract_provenance_20260925.py --readme VERIFY.md
run "imports+regions ok" env PYTHONPATH=codes:codes/paper4_nn_revision $PY -c \
  "import region_atlas, region_atlas_kit, compute_rdms, path1_confound_rsa; import bench.protocol as P; assert len(P.region_channels(306,'temporal_L'))==30; assert len(P.region_channels(208,'temporal_R'))==19; print('imports+regions ok')"
rm -rf bench/regenerated; find . -name __pycache__ -type d -prune -exec rm -rf {} +
echo; [ $fails -eq 0 ] && echo "ALL CHECKS PASSED" || echo "$fails CHECK(S) FAILED"
exit $fails
