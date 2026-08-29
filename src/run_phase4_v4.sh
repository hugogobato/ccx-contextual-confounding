#!/bin/bash
# Phase 4 v4 continuous re-run (D-P4.9: quadratic detrend adopted).
set -x
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CCX_WORKERS=6
L=results/raw/phase4/v4run.log

echo "=== v4 stage 1: p4cnull (36 groups) $(date)" >> $L
python3 src/run_p4_continuous.py --arm null >> $L 2>&1

echo "=== v4 stage 2: p4calt (144 groups) $(date)" >> $L
python3 src/run_p4_continuous.py --arm alt >> $L 2>&1

echo "=== v4 stage 3: p4cadv $(date)" >> $L
python3 src/run_p4_continuous.py --arm adv >> $L 2>&1

echo "=== v4 stage 4: p4realdata $(date)" >> $L
python3 src/run_p4_realdata.py >> $L 2>&1

echo "=== v4 stage 5: aggregate + Gate D $(date)" >> $L
python3 src/aggregate_phase4.py >> $L 2>&1

echo "=== v4 stage 6: figures $(date)" >> $L
python3 src/make_figures_phase4.py >> $L 2>&1

echo "=== V4 RUN COMPLETE $(date)" >> $L
