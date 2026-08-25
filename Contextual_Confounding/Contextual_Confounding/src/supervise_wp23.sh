#!/bin/zsh
# Crash-isolated WP 2.3 power-study supervisor (file-existence completion).
set -u
cd "$(dirname "$0")/.."
LOGDIR=/tmp/opencode
P=${CCX_WORKERS:-6}

python3 src/run_wp23_power.py --dump-groups >/dev/null || exit 1

missing() {
python3 - <<'EOF'
import json, os
groups=json.load(open("configs/phase2_power_groups.json"))
out=[i for i,g in enumerate(groups)
     if not os.path.exists(f"results/raw/phase2/wp23_{'-'.join(map(str,g['cell']))}_{g['family']}_rho{g['rho']:.1f}_n{g['n']}.csv")]
print(" ".join(map(str,out)))
EOF
}

for round in 1 2 3 4 5 6; do
  MISS=$(missing)
  if [ -z "$MISS" ]; then echo "wp23 supervisor: complete"; break; fi
  echo "wp23 round $round: $(echo $MISS | wc -w) missing"
  echo $MISS | tr ' ' '\n' | xargs -P $P -I{} sh -c \
    "python3 src/run_wp23_power.py --one {} >> $LOGDIR/wp23_one.log 2>&1"
done

MISS=$(missing)
[ -z "$MISS" ] && echo "wp23: COMPLETE" || echo "wp23: INCOMPLETE $MISS"
