#!/bin/bash
# pipeline progress: 7 training stages (4 detection + 3 classifiers)
cd /War-Damage-Assessment
started=$(grep -c "Running stage" repro.log)
echo "=== pipeline: stage $started of 7 started, $((started-1)) finished ==="
echo "--- current stage:"
grep "Running stage" repro.log | tail -1
echo "--- current epoch:"
tr '\r' '\n' < repro.log | grep -E "^ *[0-9]+/[0-9]+ " | tail -1
echo "--- finished results:"
grep -E "=== RESULT|best epoch" repro.log
