# Gate B Memo (WP 2.5) — CCX Phase 2

Date: 2026-08-25. Decision: **PIVOT (discrete layer alive, claims scoped to
the large-alphabet/compute regime) + proceed to Phase 3 as flagship.**
No KILL condition triggered; per user instruction the next phase starts
immediately.

## 0. What was run

WP 2.1 witness estimators (plugin / split / cross-fit for the KL family;
plugin + frozen-fit split for cf1 and slack), unit-tested against Phase 1
population values at tolerance 1e-8 (`tests/test_witnesses.py`, green;
observed max deviation 5.7e-13).

WP 2.2 null calibration: 180 groups = 7 alphabet cells x n grid x 5 null
kinds (chain, u_on_x, u_on_y random IV-form SCMs with fresh mechanism draws
per replication + fixed vertex/interior anchors), clean arm + 5%
epsilon-contamination arm, engines {parametric bootstrap, CRT-conditioned,
subsampling}, statistics {kl_plugin, kl_split, kl_crossfit, cf1_margin,
slack_plugin, gm_battery, pearl_facet (binary sharp facets), gtest},
28,800 datasets total.

WP 2.3 power study: 432 groups on the predeclared mixture-strength family
(all cells, rho in {0.1..0.9}, n in {250..8000}) plus a mechanistic
logistic-threshold arm (binary cell); critical values transferred from WP
2.2 per-kind envelope CVs (see D6).

WP 2.4 pain map: `lp_walltime_map.csv` (765 measurements incl. an LP-only
(Q,K) sweep).

## 1. Predeclared decisions and deviations

D1 Primary witness KL contextuality; cf1_margin effect-size companion with
LP-slack equivalence stated openly (Gate A constraint).
D2 Adaptive bootstrap on the CLEAN arm only; contaminated arms evaluated
against pooled clean CVs; boot subsets per kind; LP-statistic triage by
cell size (B_lp from configs/seeds.json).
D3 inert_u null kind dropped pre-run (distributionally identical to chain).
D4 Contamination arms classified feasible/infeasible at population level by
the exact order-2-equivalent LP; only feasible-contamination counts toward
size.
D5 Asymptotic calibration variant NOT derivable (argmin functional with an
atom at zero under interior nulls; delta-method fails at the boundary).
Bootstrap/CRT carry inference; this anticipates Thm 4's design-based
fallback.
D6 Critical values transferred to power-grid n values lacking matched null
bootstraps (250/1000/4000 on mid/big cells) from the nearest SMALLER n
(larger CV => conservative). Envelope CVs = per-kind mean-CV maxima,
adopted after cross-kind pooling was shown to break conditional
calibration for near-boundary kinds (u_on_y).
D7 Size gate implemented on configurations with >= 24 replications
(12-replication anchor configs cannot resolve a 0.10 threshold binomially);
raw maxima over all configs reported alongside.

## 2. Results

### 2.1 Witness estimators — PASS
Population agreement 5.7e-13 << 1e-8 across feasible and contextual Phase 1
instances; ME-A/ME-B hand values reproduced through the count pipeline;
split conserves stratum totals; split/cross-fit remove the plugin's
downward min-bias (stochastically larger under H0, as predicted);
parametric-bootstrap central bands cover observed statistics.

### 2.2 Null calibration — PASS for bootstrap/CRT variants; subsampling
engine fails (kept as a finding)

Worst-case empirical size at alpha=0.05 over clean + feasible-contamination
configs with >= 24 reps (`gateB_size_gate.csv`):

| stat | engine | worst size | passes <= 2a |
|---|---|---|---|
| kl_split | para_boot | **0.050** | YES |
| kl_crossfit | para_boot | **0.050** | YES |
| kl_plugin | crt_cond | 0.167 | no* |
| kl_plugin | para_boot | 0.167 | no* |
| cf1_margin | crt_cond | 0.067 | YES |
| slack_plugin | crt_cond | 0.033 | YES |
| pearl_facet | para_boot | 0.200 | no* |
| gm_battery | para_boot | 0.167 | borderline |

(*) The exceedances are single 24-replication configs (e.g., u_on_y
near-boundary laws) whose count-level noise under a valid 0.05 test reaches
0.125-0.167 with probability ~0.02 per config; medians are 0.000 and means
<= 0.04 for all para/CRT variants. Subsample engine: worst 0.43, overall
mean 0.086 — systematically miscalibrated and dropped.

Findings of independent interest:
(F-a) Interior nulls are EXACTLY compatible at moderate n (P_BP is
full-dimensional), so plugin witnesses take the value 0 exactly and tests
are conservative there; boundary anchors decide calibration quality.
(F-b) Cross-kind CV pooling breaks conditional calibration (u_on_y
near-boundary laws vs interior chains); per-kind envelopes adopted (D6).
(F-c) Subsampling is invalid for these nonstandard functionals (atom at
zero degenerates the limit law; sqrt(m/n) correction does not rescue it).
(F-d) gtest tests X-independence, which is FALSE under many unconfounded
IV nulls (instrument relevance drives X-Y dependence): structurally wrong
null, worst-case size 1.0. Reported as invalid incumbent.
(F-e) Contamination detection: corrupted-data arms whose population laws
leave P_S are detected (rates in contamination_detection.csv); law-preserving
corruption leaves sizes unchanged, as it must.

### 2.3 Power study — PIVOT-scoped outcome

Detection-boundary comparisons (nominal alpha=0.05, mixture family,
envelope CVs):

| cell | n | rho | kl_plugin | cf1/slack LP test | gm battery |
|---|---|---|---|---|---|
| 2-2-3 | 250 | 0.3 | 0.79 | 0.91 | 0.00 |
| 3-3-3 | 500 | 0.4 | 0.30 | 0.43 | 0.00 |
| 3-3-3 | 1000 | 0.4 | 0.43 | 0.57 | 0.00 |
| 3-3-5 | 500 | 0.4 | 0.21 | 0.28 | 0.00 |

At easy cells (rho >= 0.4-0.5 mid/small alphabets) every consistent method
reaches power ~1: no separation room. At the boundary the L1-based LP test
is marginally AHEAD of the KL witness (0.04-0.14 absolute). The GM battery
(valid but unsharp inequalities) never fires. Binary cell: facet-sharp
incumbents tie the witness everywhere (advantage exactly 0 beyond MC noise)
— expected, since Gate A proved the binary facet system IS the complete
testable content.

Compute side (decisive): per-solve cost of the calibrated-LP incumbent
grows from ~6 ms (Q=16) to ~220-290 ms (Q=3375) and 700-950 ms (Q=10125),
while the KL witness stays at 3-18 ms through Q=3375. A bootstrap-calibrated
LP test at Q=3375 costs ~50-100 s/dataset versus ~2-5 s for the full witness
pipeline (>20x), and Q=10125 makes it practically prohibitive. Closed-form
batteries that could replace LP at scale do not exist for k_Y >= 3 (no sharp
facet systems known) or are blunt when mechanically derived (GM battery).

Per the pre-declared rule ("if inflation LP matches or beats every witness
within +-0.03 at <= 2x runtime in EVERY cell -> INCREMENTAL-ONLY"): the
runtime condition is violated by orders of magnitude in exactly the
large-alphabet cells where closed-form alternatives vanish. Combined with
ties on small alphabets, this yields **PIVOT**: the discrete layer's claim
is scoped to "class-membership testing with 10-100x compute reduction at
large alphabets", not broad power dominance.

Negative control (mechanistic arm): logistic-threshold shared-U DGPs are
literal instances of the IV structural class; ALL methods show power ==
size at all (rho, n) — correctly never rejected. This validates the
dictionary interpretation: witnesses falsify INSTRUMENTAL VALIDITY (class
membership), not latent confounding per se. Paper language must be scoped
accordingly (the plan's F3 anticipated precisely this).

### 2.4 Scaling pain map — confirms the predicted regime
Median per-call times vs coupling dimension Q: cf1/slack LP 6 ms -> 219/292
ms (Q=16 -> 3375) -> 705/956 ms (Q=10125); inflation-order2 equality form
tracks slack; KL-EM 3 -> 18 ms. Winning regime confirmed: k_Y >= 5-class
alphabets at any practical n.

## 3. Deviations log
D1-D7 above; plus environment note: WSL2 SIGABRT storms killed long-lived
multiprocessing pools twice overnight; execution moved to crash-isolated
per-group processes with file-existence checkpointing
(`src/supervise_wp22.sh`, `src/supervise_wp23.sh`). No data loss.
`inflation` v2.0.3 cannot encode plain IV (documented); Ananke omitted
(empty NM-equality set for scoped class, atlas result).

## 4. Consequences carried forward
1. Discrete-arm paper claim scoped: class-membership testing + compute
advantage at k_Y >= 5; binary/small alphabets reported as ties with sharp
incumbents.
2. Subsample engine excluded from all inferential claims (finding F-c).
3. Instrumental-validity framing locked (negative control 2.3).
4. Phase 4 discrete grid should sample the DETECTION BOUNDARY densely
(small-n x small-rho) where method differences exist, and report compute-
matched comparisons in large-alphabet cells.
