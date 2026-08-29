# Gate D Memo (Phase 4) — CCX full falsification ladder

Date: 2026-08-29 (v4, final). Decision: **GO to Phase 5 (continuous
arm, k1, v4 variant)** — every predeclared Gate D condition is met:
region share 25% (36/144 cells, the entire conf_nonlin/t3 column,
n = 2000..20000), worst adversarial FPR 0.02 <= 0.10, sizes <= 0.025,
power monotone in n at 0.92 / 0.995 / 1.00. The earlier PIVOT
recommendation was based on the linear-detrend pipeline whose B4
adversarial FPR (0.60) exposed a scope boundary; the predeclared remedy
ablation (D-P4.9, 100 seeds) showed the per-context QUADRATIC detrend
fixes B4 (k1 0.60 -> 0.01 at 100 seeds, 0.02 at the full 200-seed rerun)
at zero cost to sizes or power, and the full continuous arm was re-run
under v4. The discrete arm keeps its Gate B positioning (compute
advantage, not power dominance) with adversarial FPR 0.00.

## Design under test

Predeclared matrix (configs/phase4.json, committed before runs) with
deviations D-P4.1..D-P4.8 logged in that file. Arms:

1. Discrete grid: (kz,kx)=(2,2), k_Y in {4,5,8} x n {500,1000,2000} x
   rho {0.2,0.4,0.6,0.8} x tail {none, t3-as-boundary-stress}; 72 cells x
   200 reps; witness family (kl_plugin/kl_split/kl_crossfit) vs
   calibrated-LP baseline (cf1_margin/slack_plugin) at matched size via
   envelope CVs (Gate B D6 rule). Engines {para_boot, crt_cond}.
2. Continuous grid: d {2,3,5} x n {2000,8000,20000} x b {0.2,0.4,0.6,0.8}
   x noise {gauss,t3}; k1/k2 vs residual-HSIC. n=2000 rows bit-identical
   reuse of verified WP 3.3 data (D-P4.2); n in {8000,20000} re-run under
   the D13-fixed code (D-P4.8) with matched nulls run locally overnight
   (D-P4.5: local execution chosen over Colab per user instruction; the
   generated `scripts/make_colab_shards_p4.py` was NOT needed).
3. Adversarial nulls: A1-A4 (discrete) + B1-B4 (continuous), 200 reps,
   per-dataset self-calibration for the continuous side.
4. Scaling study + real-data smoke (card, mroz, jtrain-1988 RCT negative
   control, card placebo).

## D13 repair (decisive engineering finding)

The Phase 4 pilots exposed an implementation defect in the frozen v3
witness: K1Witness computed b_c over up to 600 residuals but
self_mmd_const over the first 400, an internally inconsistent QP active
whenever strata exceed 600 rows (n >= 4000 at K = 8). Symptom in the
verified Gate C data: k1 power on conf_nonlin/t3 collapses 0.93 (n=2000)
-> 0.205 (n=8000). The aligned fix (m_cap=400) is a bit-identical no-op
for every Gate C verdict-relevant cell (regression vs stored wp33 rows
<= 2.5e-16) and restores power 0.72-0.88 at n in {8000, 20000}. All
n >= 4000 Phase 4 cells ran under the fixed code; pre-D13 raw rows were
moved to results/raw/superseded/phase4_preD13/. Evidence chain:
configs/phase4.json D-P4.8, memos/gate_C_memo.md addendum, regression
script output (5 stored rows reproduced exactly).

## Results (v4, final)

### Continuous arm (flagship) — GO

k1 on conf_nonlin/t3 under v4 (D13-aligned caps + quadratic detrend;
results/phase4_grid/continuous_separation.csv, 200 reps/cell):

| n | power (all d, all b) | null size (worst) | hsic power |
|---|---|---|---|
| 2000 | 0.92 | 0.005 | 0.060-0.095 |
| 8000 | 0.995 | 0.010 | 0.035-0.065 |
| 20000 | 1.00 | 0.025 | 0.045-0.050 |

All 36 conf_nonlin/t3 cells qualify (power - hsic >= 0.20, size <= 0.10):
region share 36/144 = 25% >= 20%. Power is flat in b because the
confounding channel sin(1.5U) + 0.5U is b-independent after detrending
removes the b*x term. Monotonicity: all panels pass.

Adversarial FPR at alpha = 0.05 (200 reps per config):
B1 selection-truncation 0.00, B2 mixture-X 0.00, B3 clipped-noise 0.00,
**B4 heavy-curvature 0.02** — the L4 remedy resolved the only GO blocker.
The linear-vs-quadratic detrend ablation that motivated v4 is preserved
as results/raw/phase4/p4detrend_*.csv (100 seeds; linear: B4 0.60 k1;
quadratic: B4 0.01, sizes <= 0.02 everywhere, power unchanged).

k2: 12 qualifying cells (8.3%), worst adversarial FPR 0.15 (B4 0.15,
B2 0.115; known L2 t3/mixture inflation). k2 remains the secondary
construction / mechanism-ablation column with declared limitations.

### Discrete arm — confirms Gate B positioning

0 of 72 cells show witness advantage >= 0.20 over the calibrated LP test
at matched size (LP ties or leads on power, as Gate B found); the
discrete layer's claim stays scoped to class-membership testing with a
10-100x compute advantage at large alphabets. Adversarial FPR: 0.000
across A1-A4 and all stats, including the shared-p_z observable null-
mixture (A1) and the bursty-context null (A3). Worst-case null sizes at
alpha 0.05 across the 72-cell grid stay within 2*alpha for the
bootstrap/CRT engines (subsample engine remains excluded, F-c).

### Scaling (results/phase4_grid/scaling_phase4.csv)

Point statistics at d=2: k1_v3 ~0.18 s at n=20000 (0.19 s at 50000),
k2_v3 0.045 s at 20000, hsic 0.03 s (cap 800). Full k1 permutation-
calibrated pipeline: ~12-18 s/dataset at n=20000, B=199.

### Real-data smoke (results/phase4_grid/realdata_smoke.csv, v4)

1 of 34 tests rejects: the documented mroz kl_crossfit sparse-multinomial
artifact (own-bootstrap CVs are similarly huge). card (IV): silent.
jtrain-1988 RCT negative control: silent, as required. card placebo:
silent, as required.

## Gate D decision

Per the predeclared rules (configs/phase4.json gate_D_rules), evaluated
on results/phase4_grid/gateD_evaluation.csv (v4):

- **GO to Phase 5** (continuous arm): size-controlled power advantage
  >= 0.20 over the strongest applicable baseline (residual-HSIC) across
  a contiguous region covering 25% of the arm's grid (the full
  conf_nonlin heavy-tailed-noise column, n = 2000..20000, all d),
  monotone in n and non-decreasing in b, adversarial FPR 0.02 <= 0.10,
  with the mechanism explanation carried by the k1-vs-k2 and
  detrend-order ablations (the witness reads shape/skew channels of the
  latent common cause that survive per-context standardization; the
  baseline reads only mean dependence, which the conf_nonlin channel
  hides under heavy tails).
- The discrete arm is NOT the GO arm (0 qualifying cells on power vs the
  calibrated LP test); it enters the paper as the class-membership +
  compute-advantage layer (Gate B PIVOT scope stands).
- Scope statement for the paper: the continuous witness claims operate
  within the shared-shape ANM class as measured through per-context
  quadratic detrending; B4 documents that the linear-detrend variant has
  a narrower reach (rho 0.49 vs real-data rho 0.10-0.18), which is
  preserved as an ablation rather than a limitation of the adopted
  variant.
- Phase 5 inherits: Thm 4 (CRT/bootstrap validity) and Thm 5 (continuous
  witness validity) must be stated for the v4 estimator (m_cap = 400,
  quadratic detrend, trim 0.05, label-permutation calibration).
