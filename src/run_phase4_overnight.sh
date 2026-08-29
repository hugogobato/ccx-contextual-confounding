#!/bin/bash
# Phase 4 overnight local run (user asleep; D-P4.8 fixed code).
# Stages: p4cadv re-run -> p4cnull full (locally, per user instruction)
#         -> p4realdata re-run -> aggregate -> gate D.
set -x
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CCX_WORKERS=6

echo "=== stage 1: p4cadv (fixed code) $(date)" >> results/raw/phase4/overnight.log
python3 src/run_p4_continuous.py --arm adv >> results/raw/phase4/overnight.log 2>&1

echo "=== stage 2: p4cnull full local $(date)" >> results/raw/phase4/overnight.log
python3 src/run_p4_continuous.py --arm null >> results/raw/phase4/overnight.log 2>&1

echo "=== stage 3: p4realdata re-run (D13 affects card/mroz strata > 400) $(date)" >> results/raw/phase4/overnight.log
python3 src/run_p4_realdata.py >> results/raw/phase4/overnight.log 2>&1

echo "=== stage 4: aggregate + Gate D $(date)" >> results/raw/phase4/overnight.log
python3 src/aggregate_phase4.py >> results/raw/phase4/overnight.log 2>&1

echo "=== stage 5: figures $(date)" >> results/raw/phase4/overnight.log
python3 src/make_figures_phase4.py >> results/raw/phase4/overnight.log 2>&1

echo "=== OVERNIGHT COMPLETE $(date)" >> results/raw/phase4/overnight.log
