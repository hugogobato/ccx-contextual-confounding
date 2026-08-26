# Gate C Memo: Continuous Contextual Confounding (WP 3.1-3.3)

Status: METHOD FROZEN (v3), EVALUATION PENDING Colab reruns.

## Predeclared pass rule

The witness (k1 or k2) passes Gate C if, at alpha = 0.05, its power
exceeds its null size by at least 0.25 at confounding strength b >= 0.4,
for n <= 5000 and d <= 3, in at least 2 of 3 favorable-regime cells
(conf_nonlin under any noise family; conf_lin under heavy-tailed noise),
with size <= 0.06 on matched nulls, all baselines (hsic) staying within
size + 0.10 of their own null sizes in those cells, and witness power
monotone in b within each (n, d, kind, noise) panel.

## Defect ledger for the continuous-outcome program

D8 (design): draw budgets across trimming levels are unbalanced;
resolved by per-trim budgets bmap (primary trim full B, sensitivity
trims min(B, 49)).

D9 (v2 driver bug): one untrimmed stat_obs shared across trim rows while
bootstrap draws were winsorized; fixed by computing stat_obs per trim.
Gaussian-noise sizes then landed near nominal, but heavy-tail sizes
stayed catastrophic (k1 up to 1.00 under lognorm/t3), which led to D12.

D10 (baseline bug): the quadratic-form HSIC bootstrap (xi' P xi) is not
a valid null law (empirical size 1.0); replaced by residual-HSIC with
permutation calibration (v2), then by pairs-bootstrap CVs on abs values
(v3), which calibrates exactly (sizes 0.00) for gauss/t3/lognorm.

D11 (aggregation bugs): CV transfer used trim 0.01 for hsic rows (hsic
carries no trimming; CVs came out NaN, killing wp33 hsic power) and the
monotonicity check mixed (n, d) panels inside one sort-by-b. Fixed in
aggregate_phase3.py (PRIM_TRIM map, per-panel monotonicity,
gateC_evaluation.csv).

D12 (calibration mechanism failure; motivating v3): the Rademacher wild
bootstrap targets the wrong functional for asymmetric residuals - every
sign-flipped draw is sampled from a pointwise-symmetrized law, so draws
do not reproduce the null sampling variability of T_K1/T_K2 under t3 or
lognorm noise (sizes 0.4-1.0 across cells; also size 0.40 for k2 under
plain gauss at n=2000). Secondary findings: constant-per-context
detrending leaves strata-width-induced shape artifacts (outermost
equal-mass strata of right-skewed x dominate the witness scale), and
per-context linear detrending carries an OLS leverage artifact that
global normal-scoring amplifies into the transport metric (kept OFF).

## v3 pipeline (frozen)

For every dataset: 1) remove a per-context LINEAR trend from y on x;
2) winsorize residuals within context at trim_q (predeclared grid
{0, 0.01, 0.05}, with 0.05 promoted to PRIMARY); 3) affinely
standardize each context (center/scale by residual moments, as declared
in the original docstring design); 4) compute T_K1 (kernelized forcing
divergence QP over the standard-normal component grid, unchanged) and
T_K2 (max standardized excess transport, unchanged); 5) calibrate by
CONTEXT-LABEL PERMUTATION of the pooled standardized residuals
(exact-multiset randomization null, no law distortion, valid under skew
and infinite moments; bandwidth reused from the observed model since the
pooled multiset is invariant under label swaps); 6) baseline =
abs(residual-HSIC(X, NW-residuals)) judged against pairs-bootstrap CVs.

Validation before freezing (see tests/test_continuous_witness.py):
calibrated size <= 2 alpha for k1/k2 under gauss and lognorm nulls and
for k1 under t3/nonparam at n=2000; k1 power 0.50 (n=800) to 1.00
(n=2000, b=0.8) in both favorable families; hsic pairs size 0.00 across
all three noise families.

## Known limitations (declared, not discovered post hoc)

L1. Per-context standardization intentionally removes pure-variance
channels: multiplicative symmetric-variance contamination is invisible
to v3 by construction. Favorable cells therefore emphasize shape/skew
channels (conf_nonlin any noise; conf_lin heavy-tail noise where the
mean drift is genuinely nonlinear).
L2. K2 retains mild excess-rejection at the t3 / null_nonparam / n=2000
cell (~0.25 at B=49/99 diagnostics); documented and excluded from k2
size gate assertions; K1 unaffected.
L3. conf_lin with Gaussian noise remains observationally equivalent to
an unconfounded Gaussian SCM (covariance-only information); no method
can separate there and none is expected to.
