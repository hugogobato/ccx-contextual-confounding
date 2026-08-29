# CCX result CSV schemas

All CSVs are processed outputs under `results/**`. Raw replication dumps live in
`results/raw/` and are append-only.

## results/phase2_discrete/null_critical_values.csv

Pooled null critical values (WP 2.2 -> WP 2.3 handoff): mean over clean-arm
bootstrap-enabled datasets of each dataset's own-draw CV, per configuration.

| column | type | meaning |
|---|---|---|
| cell | str | alphabet id `kz-kx-ky` |
| n | int | sample size |
| stat | str | kl_plugin, kl_split, kl_crossfit, cf1_margin, slack_plugin, gm_battery, pearl_facet, gtest |
| engine | str | para_boot, crt_cond, subsample |
| alpha | float | nominal level (0.01..0.20 grid) |
| cv_pooled | float | pooled upper-tail critical value |

## results/phase2_discrete/size_calibration.csv

Empirical sizes (WP 2.2 binding output). One row per
(cell, kind, arm_class, stat, engine, alpha).

| column | type | meaning |
|---|---|---|
| cell / kind / n | | config; kind in {chain, u_on_x, u_on_y, anchor_deterministic, anchor_interior} |
| arm_class | str | clean, contam_feasible (corrupted data whose population law stays in P_S: counts toward size), contam_infeasible (counts toward detection) |
| stat / engine / alpha | | as above |
| n_datasets | int | replication count contributing |
| size_adaptive | float | rejection rate vs own-draw CVs (boot-enabled rows only) |
| size_pooled | float | rejection rate vs pooled ensemble CVs (all rows) |

## results/phase2_discrete/selected_variants.csv

Predeclared variant selection summary: worst-case/median/min size per
(stat, engine) over clean configs, eligibility flags at binding alphas.

## results/phase2_discrete/contamination_detection.csv

Rejection rates on contam_infeasible arms (data-corruption detection).

## results/phase2_discrete/power_curves.csv

WP 3-free WP 2.3 primary output. One row per
(cell, family, rho, n, stat, engine, mode=pooled).

| column | type | meaning |
|---|---|---|
| family | str | mixture (primary strength family), mechanistic (binary logistic-threshold) |
| rho | float | confounded-mass strength (mixture) or coupling scale (mechanistic) |
| mode | str | pooled-CV decisions |
| size_at_0.05 | float | matching null empirical size at alpha=0.05 |
| power_at_0.05 | float | rejection rate at alpha=0.05 |
| power_at_size_0.04 / _0.06 | float | predeclared primary metric: linear interpolation of the power-vs-alpha curve at the alpha where null size crosses the target (NaN if the size curve never reaches the target) |
| n_reps | int | replications (200) |

## results/phase2_discrete/runtime_scaling.csv

Per-dataset total statistic-evaluation time by (cell, n): mean/median/max.

## results/phase2_discrete/lp_walltime_map.csv

WP 2.4 pain map: per-call wall seconds + tracemalloc peak bytes per method
(counts_build, kl_plugin_single, kl_plugin_batch100, split_crossfit,
cf1_lp, slack_lp, null_fit_em, inflation_order2, gm_battery_eval, gtest,
bootstrap_draw_paraB99) over cells x n x reps, plus a synthetic LP-only
(Q, K) sweep.

## results/raw/phase2/wp22_*.csv

Raw per-group rows (append-only). One row per (dataset-arm, stat, engine):
cell, kind, n, seed, arm, feas_contam, stat, engine, B (draws; 0 =
observed-only), stat_obs, cv_0.01..cv_0.20 (NaN for B=0),
inflation2_feasible_obs (clean arm boot rows only).

## results/raw/phase2/wp23_*.csv

Raw alternative-side rows: cell, family, rho, n, seed, dt_stats_s, stat,
engine (engine repeated for merge convenience), stat_obs.

## results/phase1_enumeration/t1_dictionary.csv

One row per enumerated observable instance.

| column | type | meaning |
|---|---|---|
| instance_id | str | stable id `<source>_<batch>_<idx>` |
| source | str | `uniform`, `boundary`, `vertex` |
| batch | int | batch index within source |
| seed | int | RNG seed used |
| e_000,e_001,... | float | observable vector, entries P(z,x,y), z-major then x then y |
| lp_status | int | HiGHS status code for primal feasibility LP |
| lp_feasible | bool | primal response-simplex feasibility decision (tolerance 1e-9) |
| facets_ok | bool | exact H-representation membership decision (tolerance 1e-9) |
| agree_c1a | bool | lp_feasible == facets_ok |
| slack_l1 | float | min L1 distance to feasible polytope (LP value) |
| borderline_band | bool | \|slack\| in [1e-9, 1e-7] (flagged, excluded from agreement stats) |
| inflation2_feasible | float/NaN | NaN unless in cross-check subsample; else 1.0/0.0 |

## results/phase1_enumeration/hierarchy_placement.csv

One row per contextual instance (lp_feasible == False).

| column | type | meaning |
|---|---|---|
| instance_id | str | as above |
| strongly_contextual | bool | no structured coupling supported on empirical supports (CF1_hard < 1) |
| cf1_hard | float | max fraction of contexts hit by a single deterministic response table |
| maximally_contextual | bool | every admissible coupling misses some support point in every context |
| cf1_soft | float | probabilistic fraction witness t* (LP) |
| degree_honest | float | honest-coupling total TV across contexts (= slack_l1 / 2 identically; kept for schema completeness) |
| degree_signed | float | CbD-style signed quasi-coupling variant (LP, normalized per context) |
| kl_contextuality | float | min_q sum_z KL(e_z \|\| pi_z(q)), capped at 999 if numerically unbounded |
| slack_l1 | float | L1 distance to polytope |

## results/phase1_enumeration/strictness_scan.csv

One row per contextual instance; NM equality constraints of the plain binary IV
model are empty by theory, so `nm_equalities_hold` is True throughout; margins
are the witness magnitudes themselves. Non-trivial strictness is tested in WP
1.4 structures where equality constraints exist.

| column | type | meaning |
|---|---|---|
| instance_id | str | as above |
| nm_equalities_hold | bool | all nested-Markov equality constraints satisfied |
| n_nm_constraints | int | number of nontrivial equality constraints tested (0 for IV) |
| margin_cf1_soft | float | 1 - cf1_soft (distance of fraction from noncontextual value 1) |
| margin_degree_honest | float | degree_honest |
| margin_kl | float | kl_contextuality |
| strict_instance | bool | any margin >= 0.05 |

## results/phase1_enumeration/witness_lp_redundancy.csv

Rank-correlation of each witness against LP slack across instances.

| column | type | meaning |
|---|---|---|
| witness | str | one of cf1_soft, degree_honest, degree_signed, kl_contextuality, cf1_hard |
| spearman_rho | float | Spearman correlation vs slack_l1 over all sampled instances |
| spearman_rho_contextual | float | same, restricted to contextual instances |
| n | int | rows used |
| monotone_injective_flag | bool | True if a strictly monotone fit achieves R^2 >= 0.9999 on log-ranks (redundancy alarm) |

## results/phase1_enumeration/structure_atlas.csv

One row per structure class with summary statistics.

| column | type | meaning |
|---|---|---|
| structure | str | id: A1_iv, A2_iv_direct, A3_proxy_collider, A4_bow |
| description | str | DAG summary |
| n_response_coords | int | dimension of response-variable simplex |
| n_observed_coords | int | dimension of observed vector |
| n_facets_nontrivial | int | count of nontrivial supporting hyperplanes found |
| n_scanned | int | instances scanned |
| n_feasible / n_contextual | int | counts |
| frac_contextual | float | |
| cf1_soft_min / mean | float | witness summaries over contextual rows |
| degree_mean | float | |
| has_nm_equality_constraints | bool | whether structure class has nontrivial nested-Markov equalities |
| witness_variation_nontrivial | bool | spread of witness values > threshold (pass rule input) |

## results/raw/phase3/wp32_*.csv

Null-calibration raw rows (Phase 3). One row per (dataset, method, trim):
kind (null_gauss/null_nonparam), noise (gauss/t3/lognorm), n, d, seed,
method (k1/k2/hsic), trim, B, stat_obs, cv_0.01..cv_0.20, dt_boot_s.
Noise kinds carry the PRIMARY trimming policy only (D8-rev).

## results/raw/phase3/wp33_*.csv

Separation-study raw rows: kind (conf_lin/conf_nonlin), b, plus as above
(trim = 0.01 primary), stat_obs per seed/method.

## results/phase3_continuous/null_calibration.csv

Empirical size at alpha=0.05 per (kind, noise, n, d, method, trim).

## results/phase3_continuous/crackle_stress.csv

Subset of null_calibration.csv with noise in {t3, lognorm}.

## results/phase3_continuous/separation_study.csv

Power vs b per (kind, noise, n, d, method) against transferred envelope CVs,
with null_size column and pass-rule summary (pass_rule_summary.csv).

---

# Phase 4 (results/raw/phase4/ + results/phase4_grid/)

## results/raw/phase4/p4dnull_*.csv

Discrete null-calibration raw rows. One row per (dataset, stat, engine):
cell (2-2-4/2-2-5/2-2-8), kind (chain/u_on_x/u_on_y), tail (none/t3 = the
predeclared boundary-stress arm, D-P4.1), n (500/1000/2000), seed (700k
stream), arm (clean), feas_contam, stat, engine (para_boot/crt_cond for
boot rows; none + B=0 for observed-only), B, stat_obs, cv_0.01..cv_0.20.

## results/raw/phase4/p4dalt_*.csv

Discrete alternative raw rows: cell, tail, family (mixture), rho
(0.2..0.8), n, seed (800k stream), dt_stats_s, stat, engine (none),
stat_obs.

## results/raw/phase4/p4dadv_*.csv

Discrete adversarial-null raw rows: cell, adv_id (A1..A4), tail_env, n,
seed (900k stream), feas_contam (population feasibility of contaminated
law; A4 only), stat, engine (none), stat_obs.

## results/raw/phase4/p4cnull_*.csv

Continuous null-calibration raw rows at n in {8000, 20000} (D-P4.8 fixed
code). Schema identical to wp32: n, d, noise (gauss/t3), kind
(null_gauss/null_nonparam), seed (400k stream), method (k1/k2/hsic),
trim, B, stat_obs, cv_0.01..cv_0.20, dt_boot_s.

## results/raw/phase4/p4calt_*.csv

Continuous alternative raw rows at n in {8000, 20000} under the D-P4.8
fixed code (n = 2000 reuses verified wp33 rows bit-identically). Schema
identical to wp33: n, d, noise, kind (conf_lin/conf_nonlin), b, seed
(500k stream), method, stat_obs.

## results/raw/phase4/p4cadv_*.csv

Continuous adversarial-null raw rows (B1..B4, self-calibrated per
dataset): adv_id, n, d, noise, seed (900k stream), method (k1/k2/hsic),
trim, B, stat_obs, cv_0.01..cv_0.20.

## results/raw/phase4/p4scale.csv

Scaling measurements: arm (discrete/continuous), cell, Q (coupling dim or
n), n, rep, method (stats_all/cf1_lp/slack_lp/k1_v3/k2_v3/hsic_resid),
seconds, peak_bytes.

## results/raw/phase4/p4realdata.csv

Real-data smoke rows: analysis (R1_card/R2_mroz/R3_jtrain_rct/
R4_card_placebo), method, trim, B, stat_obs, cv_0.01..cv_0.20 (own
parametric-bootstrap CVs for discrete rows; permutation CVs for k1/k2;
absent for hsic), cell/n_env (discrete rows), K_strata (continuous rows).

## results/phase4_grid/*.csv

Processed Phase 4 outputs (see src/aggregate_phase4.py docstring):
discrete_envelope_cvs, discrete_null_sizes, discrete_power_curves
(power/null_size/advantage per cell x tail x rho x n x stat x engine),
discrete_adversarial (FPR vs envelopes), continuous_null_sizes,
continuous_separation (power vs b over the Phase 4 grid with
power_hsic/advantage columns), continuous_adversarial, scaling_phase4,
realdata_smoke (stat_obs + reject_05_own/reject_05_env flags),
gateD_evaluation (per-arm region share, monotonicity, adversarial FPR,
GO inputs per the predeclared Gate D rules).
