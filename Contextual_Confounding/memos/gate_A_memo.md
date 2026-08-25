# Gate A Memo (WP 1.5) — CCX Phase 1

Date: 2026-08-24. Decision requested: GO / PIVOT / INCREMENTAL-ONLY / KILL for Phases 2 and 3, per plan Section 5.

## 1. Contribution under test

C1 dictionary (contextuality of the response-variable empirical model iff instrumental-inequality violation), C1b hierarchy placement, C2 strictness beyond equality constraints, F4 redundancy quantification. Companion reading: `memos/formalization_memo.md` (independently reviewed: verdict SOUND WITH MINOR FIXES; both flagged blockers patched before WP 1.4), `memos/prior_art_ledger.csv`.

## 2. Evidence tables

### 2.1 Dictionary correctness (C1a) — PASS

Full enumeration `results/phase1_enumeration/t1_dictionary.csv`: 1,120,016 instances (1,000,000 uniform Dirichlet; 100,000 boundary ray-exit near-facet; 20,000 sparse-support; 16 exact vertices). Facet system derived exactly (rational affine reduction + qhull candidates + exact snapping + sign validation): 12 facets = 6 positivity + 6 structural (4 sharp IV inequalities after normalization plus 2 equivalent forms).

| stream | n | feasible | agreement |
|---|---|---|---|
| uniform | 1,000,000 | 800,472 | 1,000,000/1,000,000 |
| boundary | 100,000 | 42,727 | 100,000/100,000 |
| sparse | 20,000 | 9,781 | 20,000/20,000 |
| vertex | 16 | 16 | 16/16 |
| **total** | **1,120,016** | | **1,120,016/1,120,016 = 100.0000% at tol 1e-9** |

Borderline band [1e-9, 1e-7]: zero instances flagged. Order-2 cross-check (Prop 2 of memo): 20,000/20,000 agreement. Independent reviewer re-derived the facet system end-to-end and confirmed completeness (four structural sharp inequalities), matching the classical sharp binary-IV set.

Kill rule check: no disagreement anywhere, so the C1 kill rule never triggers.

### 2.2 Hierarchy placement (C1b)

`hierarchy_placement.csv`: 267,020 contextual instances (23.84%). cf1_hard distribution {1.0: 266,832; 0.5: 188}. Strongly (support-)contextual instances: 188, all from the sparse-support stream with exact zeros in the observed supports. All-context support-maximality: fired on 0 instances. Finding: in the binary IV scenario the possibilistic tier is generically vacuous under full-support observations; the informative content lives entirely in the probabilistic tier (witness values). This sharpens the paper's framing: scalar witnesses are not decoration here, they are the only non-vacuous signal.

### 2.3 Strictness scan (C2)

The plain IV model has an empty nested-Markov equality-constraint set (memo Section 6; confirmed structurally by full-dimensionality of the polytope), so all margins equal witness magnitudes by construction: 172,633/267,020 contextual instances have margin >= 0.05. Pass rule met literally, but the honest reading is that strictness-with-equalities is not testable inside plain IV; the atlas was designed to probe it.

Atlas (`structure_atlas.csv`): A2 (exclusion broken) and A4 (mediated latent confounding) have VANISHING obstruction sets (25,000/25,000 random joints feasible; observationally unconstrained classes). A3 (pure collider) is exactly realizable per-instance via MILP over its 8 effective cell types; its feasible set is a finite union of polytopes (nonconvex), 50% contextual, but every factorized-null instance satisfying its implied equality Z indep Y was feasible: zero equality-constrained contextual instances at sampled density. Consequence per plan fail rule: Phases 2 to 4 scope to IV-like structures (explicitly anticipated as acceptable, narrower paper). The strictness theorem target (Thm 3) downgrades to the IV-specific statement recorded above unless a structure with nonempty NM-equality set intersected with contextual region is found later.

### 2.4 Redundancy vs LP slack (F4) — conclusive

`witness_lp_redundancy.csv`:

| witness | rho (all) | rho (contextual) | lin R² slack ~ (1−w) | verdict |
|---|---|---|---|---|
| cf1_soft | −0.819 | **−1.000000** | **1.00000** | redundant with LP slack at population level (exact linear identity within contextual region) |
| kl_contextuality | +0.746 | +0.601 | — | NOT a monotone transform of slack; carries distinct information |
| degree_signed | NaN (identically 0) | — | — | structurally vacuous: response columns affinely span the whole observable space, so every law admits a signed quasi-coupling; max |value| over 1.12M instances = 0.000e+00 |

Two load-bearing interpretive consequences. First, any population-level power claim for cf1_soft is a repackaged LP-slack claim; the discrete statistical layer must either exploit finite-sample/computational advantages or lean on the non-redundant KL witness. Second, the CbD signed-degree convention does not merely under-report here, it is identically zero: a clean differentiation point against the Contextuality-by-Default program (their measure cannot fire at all in this causal semantics).

## 3. Prior-art status

WP 1.1 ledger upgraded to E2/E3 with URL/DOI for every row; four API sweeps re-run: NO DIRECT HIT. Bibliographic corrections absorbed: Kedagni-Mourifie is Biometrika 107(3):661-675 (2020), DOI 10.1093/biomet/asaa003, no arXiv version; arXiv:1511.02823 is de Barros-Kujala-Oas (negative probabilities), not a Dzhafarov-Kujala CbD paper; AB's tiers are possibilistic/probabilistic/strong with strong iff maximal (their Prop 6.3), which our memo now reflects and adapts correctly.

## 4. Deviations and notes

1. Full-scan wall time ~2.4 h under CPU sharing (other workloads on the machine); uncontended pilot timing extrapolates to roughly 30-45 min, consistent with the plan's sub-hour rerun criterion at low contention.
2. Atlas scope decisions documented in memo Section 9 (conditional-form parametrization where root independence matters; MILP for the collider; bow-family replaced by mediated-confounder structure after the shared-latent constraint proved non-simplex).
3. The formalization memo passed independent review with two statement-level blockers, both patched (strict positivity of P(z) in Prop 2; A3 graph specification); micro-example tests including the machine-confirmed hand value KL(ME-B) = 2 log 2 all pass (`tests/test_micro.py`, 24 checks green).
4. Reproducibility: seeds frozen in `configs/seeds.json`; raw batch dumps under `results/raw/phase1/`; figures regenerate from CSVs via `src/make_figures.py`.

## 5. Decision

**GO to Phases 2 and 3**, scoped to IV-like structures, with two binding framing constraints carried forward:

1. Discrete-witness portfolio for Phase 2: primary witness KL contextuality; cf1_soft retained as interpretable effect size with its LP-slack equivalence stated openly; degree_signed dropped from comparisons except as the CbD-differentiation remark.
2. The continuous-outcome arm (Phase 3, flagship C4) proceeds unchanged; Gate A adds no obstacle to it.

Decision confirmed by the user on 2026-08-24 (option: GO as recommended). Phase 1 closes here per plan Section 11; Phase 2 (WP 2.1 witness estimators) and Phase 3 (WP 3.1 continuous constructions) are authorized to start in parallel.
