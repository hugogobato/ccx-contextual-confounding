#!/bin/zsh
# Crash-isolated WP 2.2 supervisor v2: one fresh process per group,
# N parallel, resume-safe. Completion is checked by FILE EXISTENCE
# (immune to signal-storm exit-code weirdness); missing groups are
# re-enqueued for up to ROUNDS passes.
set -u
cd "$(dirname "$0")/.."
LOGDIR=/tmp/opencode
mkdir -p $LOGDIR
P=${CCX_WORKERS:-6}

python3 src/run_wp22_calibration.py --dump-groups >/dev/null || exit 1

missing() {
python3 - <<'EOF'
import json, os
groups=json.load(open("configs/phase2_groups.json"))
out=[i for i,g in enumerate(groups)
     if not os.path.exists(f"results/raw/phase2/wp22_{'-'.join(map(str,g['cell']))}_{g['kind']}_n{g['n']}.csv")]
print(" ".join(map(str,out)))
EOF
}

for round in 1 2 3 4 5 6; do
  MISS=$(missing)
  if [ -z "$MISS" ]; then echo "supervisor: all groups complete"; break; fi
  N=$(echo $MISS | wc -w)
  echo "supervisor round $round: $N groups missing"
  echo $MISS | tr ' ' '\n' | xargs -P $P -I{} sh -c \
    "python3 src/run_wp22_calibration.py --one {} >> $LOGDIR/wp22_one.log 2>&1"
  rm -f results/raw/phase2/*.tmp 2>/dev/null
done

MISS=$(missing)
if [ -z "$MISS" ]; then echo "supervisor: COMPLETE"; else
  echo "supervisor: INCOMPLETE after all rounds, missing: $MISS"; fi
