#!/bin/zsh
# Crash-isolated Phase 3 supervisor (wp32 then wp33), file-existence based.
set -u
cd "$(dirname "$0")/.."
LOGDIR=/tmp/opencode
P=${CCX_WORKERS:-6}

run_block() {
  DRIVER=$1; GROUPS=$2; PATTERNS=$3; TAG=$4
  python3 src/$DRIVER --dump-groups >/dev/null || exit 1
  missing() {
    python3 - "$GROUPS" "$PATTERNS" <<'EOF'
import json, os, sys
groups=json.load(open(sys.argv[1]))
pats=sys.argv[2].split("|")
out=[i for i,g in enumerate(groups)
     if not any(os.path.exists(p.format(**g)) for p in pats)]
print(" ".join(map(str,out)))
EOF
  }
  for round in 1 2 3 4 5 6; do
    MISS=$(missing)
    if [ -z "$MISS" ]; then echo "$TAG: complete"; break; fi
    echo "$TAG round $round: $(echo $MISS | wc -w) missing"
    echo $MISS | tr ' ' '\n' | xargs -P $P -I{} sh -c \
      "python3 src/$DRIVER --one {} >> $LOGDIR/${TAG}_one.log 2>&1"
  done
}

run_block "run_wp32_calibration.py" configs/phase3_groups.json \
  "results/raw/phase3/wp32_n{n}_d{d}_{noise}_{kind}.csv" wp32

run_block "run_wp33_separation.py" configs/phase3_sep_groups.json \
  "results/raw/phase3/wp33_n{n}_d{d}_{noise}_{kind}_b{b:.1f}.csv" wp33

echo "phase3 supervisor done"
