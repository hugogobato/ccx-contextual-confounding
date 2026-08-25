# Research Plan — Idea 2: Confounding as Contextuality, Sheaf-Theoretic Witnesses for Latent Common Causes

Project code: **CCX** (Contextual Confounding, eXpanded B2)
Prepared: 2026-08-23
Source dossier: `Sheaf_Research_Ideas.md` (Idea 2)
Predecessor audit (mandatory companion reading): `Topological_Confounding_Detection/B2_topological_confounding_detection.md`

---

## 0. Executive verdict

| Item | Statement |
|---|---|
| Current classification | **Promising but unproven** (novelty provisionally clear after prior-art sweep; all empirical claims UNTESTED) |
| Confidence / evidence level | Medium; prior-art claims at E2-E3 (arXiv API sweep 2026-08-23), statistical claims at E0 |
| One-sentence contribution | Place latent confounding inside the Abramsky-Brandenburger contextuality hierarchy as a theorem-backed dictionary, then show that contextuality witnesses (fraction, degree, entropic, and a new kernelized continuous-outcome version) are statistically valid and practically superior test statistics for falsifying unconfoundedness in regimes where inflation LPs and nested Markov machinery hurt (large alphabets, modest samples, continuous outcomes). |
| Real contribution vs engine vs application | Contribution: the dictionary theorem plus the continuous-outcome witness. Engine: witness implementations and calibration procedures. Application: IV-style confounding diagnostics on real data (smoke level only). Decoration to cut if pressured: deep cohomology exposition beyond what the dictionary needs. |
| Strongest reason it could become a strong field paper | Nobody has built the AB-to-Pearl dictionary with valid statistical tests, and the continuous-outcome witness occupies genuinely empty space (inflation is discrete-native; sweep on 2026-08-23 found nothing). |
| Strongest reason it could fail or become incremental | Order-2 inflation LPs may already be cheap and powerful in every discrete regime worth caring about, and the continuous witness may fail calibration or power (the crackle problem, data hunger). Then only the dictionary remains, which alone is a workshop paper, not a field paper. |
| Next unresolved gate | **Gate A** (Phase 1): dictionary correctness + strictness scan + redundancy check, all by exhaustive small-scale computation. |
| Single cheapest decisive next action | WP 1.3: the binary-IV enumeration experiment (roughly one week of local CPU, no infrastructure). |

---

## 1. Idea reconstruction and claim decomposition

**Problem.** Latent common causes violate unconfoundedness, which invalidates most observational causal estimates. Existing detection tools are: equality constraints (Verma, nested Markov), inequality constraints (instrumental inequality, Bonet, Kedagni-Mourifie), and the inflation technique (Wolfe-Spekkens-Fritz; asymptotically complete per Navascues-Wolfe). All are discrete-native, LP-based, and come with no sampling theory to speak of.

**Unit of analysis.** An observed dataset of $(Z, X, Y)$ (instrument or context, treatment, outcome), possibly with continuous $Y$; replicated Monte Carlo samples from SCMs with and without latents.

**Observation scheme.** Observational data only in Phases 1 to 3; Phase 4 adds smoke-level real datasets.

**Target.** Not "detect confounding" (unidentifiable assumption-free; see Section 2, flaw F3). Target: **valid size-controlled tests of the unconfoundedness hypothesis within a declared structural class**, with power against named latent-confounder alternatives, plus interpretable witness values as effect sizes.

**Method.** Build the empirical model over response variables (Balke-Pearl / inflation formulation), view compatibility with the unconfounded structure as existence of a global section of a sheaf/presheaf of couplings over the cover of measurement/interventional contexts, and use contextuality witnesses as scalar test statistics with calibrated null distributions.

**Assumptions.** Declared structural class (e.g., binary IV model, then larger alphabets, then Gaussian/nonparametric continuous SCMs); faithfulness-type regularity for alternatives; tail restrictions where needed (crackle discipline inherited from B2).

**Intended audience.** UAI / CLeaR / Journal of Causal Inference, with the IV example leading.

### Claim decomposition

| ID | Claim | Type | Feasibility | Novelty | Importance | Evidence status | Verdict |
|---|---|---|---|---|---|---|---|
| C1 | Dictionary theorem: binary-IV contextuality iff instrumental-inequality violation; precise AB-hierarchy placement vs Balke-Pearl bound sharpness | contribution | High (adaptation of known LP completeness) | Medium-high | Medium | E2, numerically checkable in days | KEEP (prove only after Gate A) |
| C2 | Strictness: witnesses (entropic/fractional) fire on points satisfying all nested Markov equality constraints, quantifying the inequality-shaped gap; possibly a dichotomy theorem | contribution | Medium (outcome useful either way) | Medium | Medium | E0 | KEEP as scan, resolve honestly |
| C3 | Discrete statistical layer: plug-in witnesses with bootstrap/CRT validity beat inflation LP and nested Markov score tests on power per unit compute in large-alphabet, modest-n regimes | contribution + engine | Medium | Medium-high | High | E0, UNTESTED | KEEP, gated by Gate B |
| C4 | Kernelized continuous-outcome witness with multiplier/bootstrap validity, working exactly where PH-based B2 v1 provably cannot | flagship contribution | Medium-low | High | High | E0, UNTESTED | KEEP, gated by Gate C; this decides the paper |
| C5 | Full cohomological exposition (obstruction classes, H^1 story) | decoration | High | Low (bookkeeping) | Low | n/a | CUT to exposition minimum |

Load-bearing contribution: C4 (with C1 as the conceptual anchor). Load-bearing assumption: the declared structural class is scientifically relevant and the witness functional separates confounded from unconfounded laws within it. Most dangerous prior-art collision: someone in the quantum-foundations or CbD community publishing "contextuality measures with confidence intervals" repurposed to causality (checked 2026-08-23, not found; see Section 2.2). Attractive component to cut: C5.

---

## 2. Fatal-flaw certificate and verified prior art

### 2.1 Fatal-flaw preflight (Gate G0)

Status: **no material defect blocks planning**; two construction traps identified, each with a concrete fix that Gate A validates numerically before any expensive work.

| ID | Check | Finding | Status |
|---|---|---|---|
| F1 | Vacuity of the naive sheaf | If contexts are conditioning events $Z{=}z$ and sections are the observed conditionals, compatibility is trivial: the observational joint $P(z,x,y)$ already exists. The naive construction never fires. | Trap identified. Fix (locked into WP 1.2): build the empirical model over **response variables** (Balke-Pearl potential-outcome formulation), so compatibility means existence of a coupling of counterfactual responses subject to structural independence constraints. This is exactly the inflation-feasibility problem, and contextuality = infeasibility. Gate A confirms numerically that this construction is non-vacuous (it fires on known-confounded instances and stays silent on feasible ones). |
| F2 | Signaling/disturbance wrinkle | Instrument relevance means $P(X,Y \mid Z{=}z)$ varies with $z$, violating AB no-disturbance; vanilla AB contextuality is undefined/signaling. | Trap identified. Fix: the response-variable formulation of F1 sidesteps this (contexts reveal different response coordinates; inconsistent connectedness is expected and handled, in the spirit of CbD which explicitly permits inconsistent connectedness). WP 1.2 must write the hierarchy mapping (strong vs maximal contextuality vs CbD degree vs Balke-Pearl bound width) before coding. |
| F3 | Identifiability of the target | Assumption-free detection of confounding is impossible (Causal Hierarchy Theorem generically; for two Gaussians observationally equivalent confounded and unconfounded models exist, B2 section 2.2). | Scoped, not a defect: the paper claims falsification of unconfoundedness within a declared class, anchored on Shah-Peters hardness (any test has power only against a restricted alternative class, which we name explicitly). Give-up rule downstream inherits this: if the class must be shrunk to triviality to get validity, the project is dead. |
| F4 | Redundancy with complete LP machinery | Inflation is asymptotically complete, so at the population level no witness can carry information beyond the LP. | Reframed honestly (this was B2 Work Package 0's lesson): the contribution is statistical and computational (finite samples, large alphabets, continuous outcomes, interpretability), never information-theoretic at the population level. WP 1.3 measures the redundancy (rank correlation of witness vs LP slack) so the framing is evidence-based, not rhetorical. |
| F5 | Well-posedness of the continuous witness | Unknown whether a kernelized/smoothed contextuality functional is stable, calibrated, and powered. | This is precisely the Phase 3 designed experiment, with predeclared kill rules. |
| F6 | Heavy tails (crackle analogue) | B2 showed spurious topology under heavy tails; the continuous witness inherits this risk through extreme conditional estimates. | Stress test is mandatory in Phase 3 (WP 3.2) with trimming policies and a kill rule. |

### 2.2 Nearest-neighbor map (prior-art sweep 2026-08-23, arXiv API; plus dossier)

| Source | Same problem? | Same target? | Same method? | Same evidence? | Remaining gap | Direct-hit risk |
|---|---|---|---|---|---|---|
| Wolfe, Spekkens, Fritz, *The Inflation Technique* (J. Causal Inference 2019; arXiv:1609.00672) | Yes | Compatibility certification | LP over inflated scenarios | Population-level, discrete | No sampling theory, no continuous outcomes, no interpretable scalar effect size; our baseline to beat | Low (complementary) |
| Navascues, Wolfe (J. Causal Inference 2019; arXiv:1707.06476), asymptotic completeness | Yes | Sharp certification | LP hierarchy | Population-level | Confirms F4; motivates statistical-layer positioning | None |
| Nested Markov models (Richardson, Evans, Shpitser, Robins); Ananke software | Yes (latent footprints) | Equality constraints | Graphical/algebraic | Asymptotic tests exist for equalities | Misses all inequality constraints (our C2 gap); discrete-parametric tractability issues | Low |
| Pearl, *Causality* Ch. 8; Balke-Pearl bounds; Kedagni-Mourifie generalized IV inequalities; Bonet inequalities | Yes | Inequality constraints for IV-like structures | Closed-form inequalities / moment LPs | Econometric testing literature exists | Fragmented, scenario-specific; no unified witness family or hierarchy placement | Medium-low (must cite and outperform on power) |
| Abramsky, Brandenburger (NJP 2011; arXiv:1102.0264) | Foundations only | Contextuality hierarchy, entropic measures | Sheaf cohomology, LS/entropy measures | Quantum/behavioral data | No causal semantics, no SCM constraints, no econometric testing | None (we extend them) |
| Contextuality-by-Default (Dzhafarov, Kujala and school; e.g. arXiv:1511.02823 tutorial; Cervantes-Dzhafarov arXiv:1812.00105 with bootstrap reliability) | Partially (systems of contextual random variables) | Measures of contextuality with inconsistent connectedness | Quasi-coupling LPs, degrees of contextuality, bootstrap reliability in psychology | Behavioral experiments | No SCM/causal semantics, no identification-theoretic dictionary, not positioned against inflation/nested Markov | Medium (nearest living neighbor on the statistics side; differentiate explicitly in the paper) |
| Gogioso, Pinzani trilogy (arXiv:2206.08911, 2303.07148, 2303.09017) | Foundations | Causally-induced contextuality, input histories | Category/sheaf theory | Conceptual | No estimation or testing | None (bridge for interventional contexts) |
| Sargsyan, cubical topos causal models (arXiv:2607.15629, Jul 2026, found in sweep) | Foundations | Holonomy-class contextuality obstruction in topos causal models | Machine-checked formalisation | Logical, not statistical | No statistics, no estimation, no confounding-testing pipeline | Low-medium (cite; their obstruction is our C5 flavor, theirs stops at logic) |
| Vallée, Markham, sequential contextuality (arXiv:2509.14125) | Adjacent | Sequential/no-disturbance hidden variable models | Inequality translation | Conceptual + examples | Dynamic settings, not confounding testing | Low |
| D'Acunto, Di Lorenzo, Barbarossa (CAN, arXiv:2509.25236) | No (multi-agent knowledge gluing) | Consistency of Gaussian SCMs | Spectral sheaf theory | Gaussian-linear | Orthogonal | None |
| HOLOGRAPH (arXiv:2512.24478) | No (LLM beliefs) | Coherence as global section | Presheaf bookkeeping | Empirical NLP | Orthogonal | None |
| Krasnovsky (arXiv:2509.07149, transformer circuits) | No | Sheaf-consistency scores | Diagnostic scores | ML interpretability | Shows the "consistency score as diagnostic" pattern exists elsewhere; cite for framing | None |
| Entropic causal inequality line (Chaves and collaborators) [verify exact refs] | Yes | Entropic relaxations of causal constraints | Information inequalities | Population-level | No sampling theory; natural comparator for C2 | Medium-low |

Statement of novelty backed by the sweep: the **combination** of (i) a proved AB-to-Pearl dictionary, (ii) witness statistics with valid inference compared head-to-head against inflation LP and nested Markov tests, and (iii) a continuous-outcome extension, is unoccupied as of 2026-08-23. Each ingredient alone exists; the plan's job is to make the combination load-bearing rather than cosmetic, with (iii) as the flagship.

Search strings used (for the record and for Gate A re-run): `"contextuality" AND "confounding"`, `"inflation technique" AND "continuous"`, `"contextuality" AND "instrumental"`, `"contextuality-by-default" AND "statistics"` on the arXiv API; planned additions for WP 1.1: Google Scholar variants of `holonomy causal discovery`, `contextual confounder test`, `Bell inequality instrumental variable`, `entropic causal inequality test`.

### 2.3 Evidence ledger (compact)

| ID | Claim | Source | Location | Level | Implication |
|---|---|---|---|---|---|
| E-1 | No existing contextuality-based confounding test with inference | arXiv API sweep 2026-08-23 (4 queries) | feed metadata inspected | E3 (for absence-of-hit at query level) | Supports provisional novelty; re-run wider strings in WP 1.1 |
| E-2 | Topos holonomy obstruction exists, logic only | arXiv:2607.15629 abstract | abstract | E2 | Cite and differentiate |
| E-3 | CbD measures + bootstrap exist outside causality | arXiv:1511.02823, arXiv:1812.00105 | abstracts | E2 | Nearest neighbor on statistics; differentiate |
| E-4 | Inflation completeness (population level) | Navascues-Wolfe arXiv:1707.06476 | dossier + memory | E2 (inspect in WP 1.1) | Fixes F4 framing |
| E-5 | Binary-IV sharpness of instrumental inequality | Balke-Pearl 1997; Pearl Causality Ch. 8 | dossier | E2 (inspect in WP 1.1) | Makes C1 an adaptation-level proof |
| E-6 | All power/calibration statements | none yet | n/a | E0 | UNTESTED; gates below |

---

## 3. Impact thesis and skeptical-referee test

Why the problem matters: unmeasured confounding is the central threat to observational causal claims; practitioners currently have either brittle graphical tools or LP machines with no error control, and nothing at all for continuous outcomes.

What changes if CCX works: a practitioner with a moderate-sample IV-type dataset and a many-valued or continuous outcome gets a calibrated, interpretable scalar alarm ("this joint law is incompatible with every unconfounded model in the declared class, witness value 0.37") that runs in seconds where inflation LP times out, with theorems explaining exactly what fired.

Who uses/cites it: causal-inference methodologists (UAI/CLeaR/JCI crowd), the inflation community (sampling theory is their acknowledged missing piece), applied econometricians testing IV validity (Kedagni-Mourifie users), and the quantum-foundations contextuality community (first serious applied-causal deployment).

Why simpler incumbents are insufficient: nested Markov misses all inequality constraints (F4-adjacent, provable); inflation LP is complete but has no sampling theory, chokes on large alphabets and continuous outcomes, and returns feasibility bit rather than effect-size; closed-form inequalities are scenario-specific and conservative.

Why not merely a combination: the dictionary theorem is a genuine bridge (new statement, adaptation-level proof), and the continuous witness is a new object, not a wrapper.

Most damaging plausible referee paragraph: "At the population level the inflation hierarchy is complete, so your witnesses are lossy compressions of LP feasibility. In your discrete experiments the alphabets you claim are painful for inflation are handled fine by modern solvers at order 2. For continuous outcomes you abandon the combinatorial structure that makes contextuality well-defined, so what precisely is your object and in what sense is your bootstrap valid?"

Evidence that answers it: (a) runtime/memory scaling curves showing a concrete regime (alphabets, n, d) where order-2 inflation is genuinely infeasible and witnesses remain powered (Phase 2 WP 2.4); (b) power crossovers with uncertainty, not averages; (c) for continuous outcomes, a precise definition plus calibration validity under stated tail conditions, with the crackle stress test reported honestly (Phase 3).

Impact dimensions (planning-stage; UNTESTED where empirical):

| Dimension | Score | Note |
|---|---|---|
| Problem importance | 3 | Core causal-inference problem |
| Novelty after prior art | 2 | Defensible new layer (dictionary + statistical + continuous) |
| Mechanism or insight | 2 (hypothesis) | Separation mechanism must be shown by ablation |
| Empirical advantage | UNTESTED | Gate B/C |
| Applied value | UNTESTED | Smoke level only; legitimately methodology-led |
| Generality | 2 | Family of IV-like structures + continuous class |
| Credibility | 2 (by construction) | Impossibility anchors built in |
| Paper coherence | 2 | One story: contextuality as calibrated confounding falsification |

---

## 4. Dependency graph and gate map

```text
Gate A = dictionary + strictness + redundancy + prior-art re-check (Phase 1)
  |
  +--> Phase 2 (discrete engine) ----------> Gate B (calibration + power vs LP/NM)
  |                                              \
  +--> Phase 3 (continuous witness) ------> Gate C (separation + validity)  [runs after WP1.2/1.3, parallel to Phase 2]
                                                 \                            /
                                                  +--> Phase 4 (full ladder) --> Gate D
                                                                             |
                                                  Phase 5 (theory + paper) --^  DORMANT UNTIL GATE D
```

Rules: Phase 2 and Phase 3 may proceed in parallel only after Gate A passes; both sit behind the same cleared gate and neither consumes the other's output. Phase 4 starts only with at least one of B, C passed. Phase 5 starts only after Gate D. Real-data work is read-only feasibility before Gate D.

---

## 5. Phase-by-phase execution program

Conventions: every work package names exact outputs; every phase ends with a written gate memo (`memos/gate_X_memo.md`) containing the decision, evidence, deviations, and consequences. "Give up" rules are binding, not advisory.

---

### PHASE 1: Enabling formalism + exhaustive small-scale falsification

Purpose and scientific question: lock the mathematical objects (fixes F1/F2), then answer by exhaustive computation the three questions that decide everything else: does the dictionary hold (C1), do witnesses carry any strictness beyond equality constraints (C2), and are witnesses reducible to LP slack (F4 quantified). This is B2's Work Package 0 answered in the contextuality formulation.

Prerequisites: none. Duration: 2-3 weeks. Compute: laptop/local CPU only.

#### WP 1.1 Prior-art consolidation and ledger upgrade

- Objective: raise E-levels; rule out residual direct hits.
- Actions: 1. Download and read (relevant sections only): AB 2011 (sections on empirical models, hierarchy, entropic measures); Navascues-Wolfe completeness statement; Balke-Pearl 1997; Kedagni-Mourifie inequality statement; CbD measure definition paper; `inflation` package docs (github.com/ecboghiu/inflation); Ananke docs (gitlab.com/causal/ananke). 2. Run the additional Scholar/query strings listed in 2.3. 3. Write the ledger.
- Outputs: `memos/prior_art_ledger.csv`, updated nearest-neighbor table.
- Verification: every load-bearing row at E2+ with URL/DOI; direct-hit verdict written explicitly.
- Pass rule: no direct hit found. Fail rule: direct hit found, stop, escalate to user with pivot options.
- Compute: negligible. Parallel with: WP 1.2.

#### WP 1.2 Formal object lock-in memo

- Objective: write down, unambiguously, the objects of F1/F2 fixes so that code is meaningless-free.
- Contents (minimum enabling formalization only, tagged `enabling`, not paper theory): base space (set of contexts: instrument values, later do(X=x)); stalks = spaces of couplings over response variables; restriction maps = marginalization plus structural independence constraints; empirical model; global section; contextuality = empty global-section fiber; witness functionals (CF(1) fraction, CbD-style degree, CbG/relative-entropy-of-contextuality); the unconfounded null H0 per structure class; the named alternative classes.
- Outputs: `memos/formalization_memo.md` (target 3-4 pages).
- Verification: independent read-through by a sub-agent checking each definition is typed, quantified, and used; two hand-worked micro-examples (one contextual, one non-contextual binary IV instance) included and checked by hand.
- Scientific pass rule: micro-examples come out as expected. Fail rule: any definition ambiguity that changes WP 1.3 results, rewrite before coding.
- Dependencies: WP 1.1 started (can overlap).

#### WP 1.3 Binary-IV enumeration experiment (the decisive cheap one)

- Objective: settle C1, C2, F4 numerically on the binary IV scenario.
- Design: scenario Z, X, Y binary, latent U with |supp U| in {2,...,6}; ground truth compatibility = exact LP feasibility of the Balke-Pearl response-function polytope, cross-checked with order-2 inflation LP on a subsample; population-level witnesses computed exactly (small LPs/QPs).
- Sampling: uniform Dirichlet draws over the observable simplex (10^6 points, seeds 0..99 batches of 10^4), plus boundary-focused sampling near polytope facets (10^5 points), plus exact polytope vertices. Deterministic, logged.
- Checks: (C1a) contextuality iff LP-infeasibility, agreement rate with tolerance 1e-9; (C1b) hierarchy placement: classify each contextual instance as strongly vs maximally contextual; relate witness value to distance-to-feasible-set (Balke-Pearl bound width); (C2) count instances where entropic witness exceeds tolerance while all nested Markov equality constraints hold (expected: yes, near facets; record margins); (C4-redundancy) Spearman rank correlation of each witness vs LP slack (L1 distance to polytope); flag if any witness is an injective monotone transform of LP slack across all instances.
- Outputs: `results/phase1_enumeration/t1_dictionary.csv`, `hierarchy_placement.csv`, `strictness_scan.csv`, `witness_lp_redundancy.csv`, figures under `figures/phase1/`.
- Verification: mechanical, script reruns end-to-end in under 1 hour on local CPU, all seeds logged; scientific, agreement rates and margins reported per check.
- Pass rules: C1a agreement = 100 percent; C2 yields at least one strictness instance with margin >= 0.05; redundancy analysis conclusive either way.
- Give-up (kill) rules for this package: if C1a shows ANY disagreement after a one-week double-check-and-repair window (implementation bugs excluded), the dictionary claim as conceived is wrong: KILL C1, and if no repaired variant restores agreement, KILL the project (the conceptual core is broken). If redundancy shows witnesses are pure monotone transforms of LP slack AND C2 margins are all < 0.01, Gate A returns KILL unless WP 1.5 scoping finds a credible continuous story (it currently does; then PIVOT to continuous-first framing).
- Compute: < 1 hour per full rerun, local. Parallel with: nothing (blocking).

#### WP 1.4 Structure atlas expansion

- Objective: chart which 3-4 node latent structures produce nontrivial witness behavior (IV variants, proxy structures Z->X<-U->Y style, bow arcs), feeding Phase 2 scope.
- Actions: repeat WP 1.3 checks on each structure class with reduced sampling density; tabulate where contextuality is even possible (some structures have vanishing obstruction sets).
- Outputs: `results/phase1_enumeration/structure_atlas.csv`.
- Pass rule: atlas identifies at least two structure classes with non-degenerate witness variation. Fail rule: only IV works, scope Phase 2-4 to IV-like structures (acceptable, narrower paper).
- Compute: < 3 hours local. Parallel with: nothing.

#### WP 1.5 Gate A memo and decision

Contents: restated contribution under test; evidence tables; decision from the status vocabulary; consequences. Decision rules:

- **GO** to Phases 2 and 3: C1 holds, strictness instances exist, no direct prior-art hit, redundancy honestly characterized.
- **PIVOT** (to continuous-first): dictionary holds but discrete witnesses are redundant with LP slack; Phase 2 compresses to a calibration harness for the discrete sanity section; Phase 3 becomes the core.
- **INCREMENTAL-ONLY**: dictionary holds, everything else redundant, and continuous scoping looks weak; consult user on whether a methods-note venue suffices; default is stop.
- **KILL (give up on the research)**: C1 fails unrepaired, or a direct prior-art hit absorbs the contribution, or (redundancy AND no strictness AND no credible continuous story). Write `CCX_Research_Diagnostic.md` instead of continuing.

Estimated totals: 2-3 weeks, all local CPU, zero Colab need.

---

### PHASE 2: Discrete-witness statistical engine (calibration + power vs incumbents)

Purpose: does the statistical layer (C3) survive contact with finite samples and fair baselines?

Prerequisites: Gate A = GO or PIVOT-with-discrete-alive. Duration: 3-5 weeks. Compute: local CPU; Colab only if WP 2.4 cells exceed thresholds (see Section 8).

#### WP 2.1 Witness estimators

Implement plugin, split-plugin (sample-splitting across contexts), and debiased (one-step) estimators of each witness in `src/witnesses.py`; unit tests against exact population values from Phase 1 instances (tolerance 1e-8 population, Monte Carlo bands for estimators). Outputs: module + `tests/test_witnesses.py` green. Trap: optimistic bias from computing witness and calibration on the same sample; the split variant exists for this reason.

#### WP 2.2 Null calibration suite

DGPs: unconfounded structures from the atlas (no U; and U present-but-independent controls), n in {250, 500, 1000, 2000, 4000, 8000}, alpha in {0.05, 0.10}; calibration methods: parametric bootstrap, subsampling (m = n^{0.7}), conditional randomisation test (CRT conditional on Z counts), asymptotic approximation where derivable. Contamination stress: 5 percent epsilon-contaminated outcomes. 200 seeds per cell. Predeclared requirement: worst-case empirical size <= 2 alpha across all null configurations after trying all calibration variants. Outputs: `results/phase2_discrete/size_calibration.csv`, size-vs-n plots. Give-up rule: if NO calibration variant holds worst-case size <= 2 alpha, the discrete statistical layer is dead (KILL C3; pivot everything on Phase 3).

#### WP 2.3 Power study vs incumbents

Alternatives: latent U strength rho in {0.1,...,0.9}; alphabets (k_Z, k_X, k_Y) in {(2,2,2),(2,2,3),(2,3,3),(3,3,3),(2,2,5),(2,2,8),(3,3,5)}; n as in WP 2.2. Competitors (equal information, same data): order-2 inflation LP (`inflation` package, runtime capped at 10 minutes per instance then logged as timeout-fail), nested Markov score test (Ananke), Pearl instrumental inequality test, Kedagni-Mourifie generalized IV test, Bonet inequalities, plain G-test of independence. Primary metric (declared before running): power at matched empirical size in {0.04, 0.06}; secondary: runtime, memory, witness interpretability. Outputs: `results/phase2_discrete/power_curves.csv`, `runtime_scaling.csv`. Give-up rule: if inflation LP matches or beats every witness within +/-0.03 absolute power at <= 2x runtime in EVERY (n, k, rho) cell tested, C3 is INCREMENTAL-ONLY at best; combined with a Phase 3 failure this terminates the project.

#### WP 2.4 Scaling pain map

Measure LP runtime/memory as functions of alphabet product and scenario size; locate the concrete regime where witnesses win on compute (predicted: k_Y >= 5 with n <= 2000; verify or refute). Outputs: `results/phase2_discrete/lp_walltime_map.csv`. This produces the referee-answer (a) in Section 3.

#### WP 2.5 Gate B memo

- **GO** to Phase 4 (discrete arm): size controlled and a non-empty, non-contrived winning region (>= 3 adjacent cells with power advantage >= 0.15 absolute, robust to seed perturbation).
- **PIVOT**: winning region exists only in large-alphabet cells; scope paper claims to that regime explicitly.
- **INCREMENTAL-ONLY**: wins exist but are tiny/fragile (see WP 2.3 give-up rule); merge as secondary evidence behind Phase 3.
- **KILL (give up)**: calibration impossible, or no winning region anywhere.

Colab trigger: any WP 2.3 cell family estimated > 2 hours or > 4 GB (measure via WP 2.4 pilot first); expected: mostly unnecessary, except possibly k_Y = 8 inflation-LP baselines, which can be sharded (Section 8).

---

### PHASE 3: Continuous-outcome kernelized witness (flagship)

Purpose: the genuinely novel territory (C4). Does a smoothed witness separate confounded from unconfounded laws exactly where support-topology methods provably fail (B2 section 2.2), with valid calibration?

Prerequisites: Gate A = GO; WP 1.2 memo. Duration: 4-6 weeks. Compute: local prototyping; Colab for OT/MMD-heavy sweeps.

#### WP 3.1 Define and implement two candidate constructions (pick both, then prune to one by WP 3.3)

Construction K1 (kernelized empirical model): discretize contexts via instrument values or fixed do(X = x) levels; smooth conditional laws with bandwidth h; apply discrete witnesses across an h-grid; stability diagnostic across h (witness curve flattening) doubles as a tuning-robustness check. Construction K2 (transport defect): compose conditional transport maps across the context cover and measure closure defect (MMD or W2 distance), the continuous descendant of B2's monodromy statistic but operating on conditional families directly, requiring no support holes and thus immune to the B2 section 2.2 failure mode. Implementation deps: POT or ott-jax, giotto-tda not required. Outputs: `src/continuous_witness.py`, `tests/test_continuous_witness.py`.

#### WP 3.2 Calibration and crackle stress

Null: unconfounded Gaussian and nonparametric SCMs, n in {500, 2000, 8000}, d in {2, 3, 5}; calibration: multiplier bootstrap, wild bootstrap, CRT conditional on X/Z; heavy-tail stress: t3 and lognormal noises, with trimming/Winsorizing policies predeclared (e.g., trim at 1st/99th percentile, report sensitivity to the trimming quantile). Requirement: worst-case size <= 2 alpha within the declared tail class after trimming. Outputs: `results/phase3_continuous/null_calibration.csv`, `crackle_stress.csv`. Give-up rule: if size cannot be controlled within the declared class (including trimmed versions), restrict scope once; if still failing, KILL C4 and fall back per Gate C rules.

#### WP 3.3 Decisive separation study (B2 section 2.2 regime)

DGPs: X = U + eps_X, Y = b*X + U + eps_Y (support of (X,Y) is R^2, contractible; persistent homology blind by construction), plus nonlinear/mechanism-drift variants; b (confounder coefficient) in {0.1,...,0.9}; d extended with nuisance covariates; n in {500,...,20000}. Baselines: HSIC/KCI conditional-independence tests, knockoff-CRT with non-topological statistics (knockpy), Janzing-Scholkopf concentration heuristic, FCI/RCD where applicable, and (where a plausible proxy exists) proximal/negative-control methods. Predeclared pass rule: witness power >= size + 0.25 at b >= 0.4, n <= 5000, d <= 3, with size <= 0.06 at alpha = 0.05, in at least 2 of 3 favorable-regime cells where ALL baselines remain <= size + 0.10, and power is monotone increasing in b. Outputs: `results/phase3_continuous/separation_study.csv`, ablation table (K1 vs K2, with/without smoothing, with/without trimming). Give-up rules: if witness power - size < 0.15 uniformly across the favorable sweep while some baseline reaches >= 0.30, KILL (adds nothing); if the pass rule can only be met with n > 10^6 at d = 2, KILL (data hunger is fatal); if separation exists but plain HSIC-CRT matches it everywhere, INCREMENTAL-ONLY (decide against the user's bar; default stop).

#### WP 3.4 Gate C memo

- **GO** to Phase 4 (continuous arm): pass rule met.
- **PIVOT**: witness works in a narrower but real sub-region (e.g., only d = 2-3, sub-Gaussian tails); scope accordingly and proceed.
- **INCREMENTAL-ONLY / KILL (give up)**: per WP 3.3. On KILL, the project reverts to B2's safe fallback (topological specification testing, B2 WP-C) or terminates; consult user.

---

### PHASE 4: Full falsification ladder at scale (shard-ready)

Purpose: convert isolated wins into a defensible engine: crossover grids, dimension and sample sweeps, adversarial nulls, scaling, real-data smoke. Prerequisites: Gate B and/or Gate C = GO/PIVOT. Duration: 4-6 weeks elapsed (wall-clock dominated by sharding logistics). Compute: local pilots, then Colab sharding per Section 8.

Experiment matrix (staged; every cell 200 replications, seeds from the global registry):

| Block | Factors | Cells | Shards |
|---|---|---|---|
| Discrete grid | k_Y in {4,5,8} x n in {500,1000,2000} x rho in {0.2,0.4,0.6,0.8} x tail in {none,t3} | 72 | 12 |
| Continuous grid | d in {2,3,5} x n in {2000,8000,20000} x b in {0.2,0.4,0.6,0.8} x noise in {gauss,t3} | 72 | 16 |
| Adversarial nulls | selection-induced holes, mixture nulls, censoring (continuous); deterministic-nonlinear unconfounded (discrete) | 8 configs | 4 |
| Scaling study | runtime/memory vs (n, d, k) for witnesses and all baselines | sweep | 4 |
| Real-data smoke | 2 public IV-type datasets (e.g., quarter-of-birth subsample, 401k-style IV data) as demonstration; RCT-derived negative controls where unconfoundedness holds by construction | 4 analyses | 4 |

Total: 40 shards maximum, matching the notebook budget exactly. Procedure: run ONE pilot shard per block locally, measure wall time and peak memory, then offload blocks whose projected per-shard cost exceeds 2 hours or 4 GB (Section 8 has the notebook template). Predeclared primary comparison per block: witness family vs strongest applicable baseline at matched size, with uncertainty across replications.

Gate D decision rules:

- **GO** to Phase 5: at least one arm (discrete or continuous) exhibits size-controlled power advantage >= 0.20 absolute over the strongest applicable baseline across a contiguous region covering >= 20 percent of that arm's grid, explained by the mechanism ablation, with adversarial false-positive rate <= 10 percent.
- **PIVOT**: advantage real but confined to a narrower declarable regime; paper scoped to it.
- **INCREMENTAL-ONLY**: wins below those thresholds or fragile (survive < 80 percent of seed-perturbation reruns); consult user; default terminate.
- **KILL (give up)**: no regime anywhere satisfies size control + meaningful advantage; or adversarial false positives > 15 percent after all fixes; or scaling makes target regimes infeasible even sharded. Write the negative-result note; revert or stop per Gate C fallback.

---

### PHASE 5: Evidence-earned theory + paper consolidation

**DORMANT UNTIL GATE D.** Purpose: prove only what survived, secure the claims the paper actually makes. Duration: 6-10 weeks.

Theory target table (tags: direct / adaptation / conjecture; each with numerical falsifier already produced upstream):

| Target | Why it earned cost | Sketch | Tag | Source results to lean on | Adaptation gap | Falsifier | Stop rule |
|---|---|---|---|---|---|---|---|
| Thm 1 (Dictionary, binary IV): contextuality of the response-variable empirical model iff instrumental-inequality violation | Gate A C1 | LP duality: facets of the Balke-Pearl coupling polytope = AB coupling constraints | adaptation | Pearl Causality Ch. 8; Balke-Pearl 1997; AB 2011 empirical-model machinery | Translate facets into AB hierarchy language | Phase 1 `t1_dictionary.csv` (100 percent agreement) | Attempt 2 weeks; if stuck, ship as proposition with exhaustive numerical verification and honest tag |
| Thm 2 (Hierarchy placement): relation of strong/maximal contextuality and CbD degree to Balke-Pearl bound width | Gate A C1b | Witness value bounds distance-to-feasibility | adaptation/conjecture | AB 2011; Dzhafarov-Kujala degree definitions | Cross-framework definitions differ; needs a bridging lemma | `hierarchy_placement.csv` | Downgrade to conjecture with numerics if the bridging lemma resists |
| Thm 3 (Strictness): witnesses fire beyond the nested-Markov equality cone; possibly a dichotomy for finite alphabets | Gate A C2 | Cone-comparison argument | conjecture (with numerics) | Chaves-style entropic inequalities [verify refs]; nested Markov constraint characterization | Characterizing the NM cone generally is hard | `strictness_scan.csv` | Ship empirical strictness section if theorem resists |
| Thm 4 (Validity): asymptotic normality / bootstrap consistency of plugin witnesses under H0, finite alphabets | Needed to interpret Phase 2 inference | Empirical-process / delta-method on multinomial functionals | adaptation | Standard M-estimation; CbD bootstrap practice (Cervantes-Dzhafarov) | Witnesses are argmax/optimization-valued; handle via Danskin-type arguments | WP 2.2 calibration curves | If optimization-valued analysis stalls, switch inference to CRT/permutation validity (design-based, weaker assumptions) and drop asymptotics |
| Thm 5 (Multiplier/wild bootstrap validity for the continuous witness) | Needed for the flagship's error bars | Kernel-statistic multiplier bootstrap | conjecture | Multiplier-bootstrap theory for kernel statistics; Shah-Peters GCM framework as template | Nonstandard functional (optimization over couplings of smoothed laws) | WP 3.2 calibration MC | Same fallback as Thm 4 |

Ordering: Thm 1, then Thm 4 (or its CRT replacement), then Thm 2, Thm 3, Thm 5. Every target carries its stop rule; none blocks submission if the design-based fallback covers the paper's inferential claims.

Paper skeleton: lead with IV example + dictionary; power/runtime tables; continuous witness as flagship (conditional on Gate C); impossibility anchors and scope statements prominent (Shah-Peters framing); venues UAI / CLeaR / Journal of Causal Inference. Cut C5 cohomology exposition to one section. Referee simulation pass + reproducibility audit before any submission decision (Gate E memo).

Give-up rule for Phase 5 (not a project kill): if any theorem exceeds its stop rule, downgrade the tag, adjust the paper's claims, and move on; never let proof difficulty inflate the claimed contribution.

---

## 6. Claims-to-experiments matrix (decisive entries)

| Claim | Mechanism | DGP | Metric | Baseline | Ablation | Threshold (predeclared) | Falsifier | Output |
|---|---|---|---|---|---|---|---|---|
| C1 dictionary | Coupling-polytope equivalence | Binary IV enumeration | Agreement rate | Balke-Pearl LP (oracle) | none needed | 100 percent within 1e-9 | Any counterexample instance | `t1_dictionary.csv` |
| C2 strictness | Inequality-shaped gap vs equality cones | Facet-focused sampling | Strictness incidence + margin | Nested Markov equality set | entropic vs fractional witnesses | >= 1 instance, margin >= 0.05 | All-zero scan | `strictness_scan.csv` |
| C3 discrete power | Witnesses cheaper/stabler than LP at finite n | Alphabet x n x rho grid | Power at matched size | Order-2 inflation LP; NM score test; KM test | split vs plugin estimator | >= 0.15 abs advantage in >= 3 adjacent cells | LP dominates everywhere | `power_curves.csv` |
| C4 continuous power | Smoothed witness sees conditional-law obstruction, not support geometry | B2 section 2.2 linear-Gaussian sweep | Power at matched size | HSIC/KCI, knockoff-CRT, JS heuristic | K1 vs K2; trimming on/off | Pass rule in WP 3.3 | Baselines match or witness blind | `separation_study.csv` |
| Validity | Calibration machinery | Null DGPs + heavy tails | Worst-case size | n/a (all methods must pass) | calibration variants | <= 2 alpha | Persistent over-size | `size_calibration.csv`, `null_calibration.csv` |
| Scaling | Computational advantage | (n, d, k) sweeps | Wall time, memory | inflation LP | solver variants | concrete winning regime exists | LP cheap everywhere | `lp_walltime_map.csv` |

Every analytic prediction above (agreement rates, monotonicity, size levels) has the numerical estimator and tolerance named. Any claim lacking a live falsifier at gate time is cut from the paper.

---

## 7. Fair-comparison protocol

Shared across all phases: identical simulated datasets fed to every method within a replication (same seed stream); identical preprocessing; tuning budgets fixed and equalized (bandwidth grids of the same size as competitor smoothing parameters; inflation order fixed at 2, with order-3 spot-checks on a subsample reported separately); timeouts logged as failures, never silently dropped; failed runs preserved with reasons in `results/**/failures.csv`. Seeds: master registry `configs/seeds.json`; every figure/table regenerable by `make results` (Section 10). Report per-cell: median and IQR across 200 replications, worst-regime value, runtime, peak memory. Primary metric declared per block before running (Section 6); all others secondary.

Oracle/diagnostic variants (true-U-based tests, LP-at-order-3) labeled DIAGNOSTIC and excluded from go decisions.

---

## 8. Compute and Colab sharding policy

Hardware reality: local machine is a 13th-gen i9-13900H (10 physical cores / 20 threads) shared with other experiments; treat 12 worker processes and an 8 GB local memory budget as the ceiling, one BLAS thread per worker, no nested parallelism.

Offload rule (binding): any experiment (cell family or single long run) projected to exceed **2 hours wall time or 4 GB peak RAM** runs as Google Colab notebooks instead of locally. Budget: **up to 40 independent, self-contained notebooks**, allocated per the Phase 4 table (12 + 16 + 4 + 4 + 4). Rationale: each Colab account provides roughly 2 cores and ~12 GB RAM (optionally a small GPU, not needed here), notebooks run up to ~10 hours, and they run independently, so embarrassingly parallel Monte Carlo shards map onto notebooks one-to-one.

Notebook template requirements (enforced by `colab/templates/ccx_shard_template.ipynb`):

1. Fully self-contained: embeds the DGP, witness, and baseline code inline (no pip installs beyond numpy/scipy/pandas/pot, pinned versions); takes SHARD_ID, SEED_LIST, and CONFIG_JSON from the first cell.
2. Appends results incrementally to `<block>_shard{SHARD_ID:02d}.csv` so a timeout loses at most the current replication.
3. Ends every output file with the safe download fallback:

```python
try:
    from google.colab import files
    files.download(output_file)
    print("Downloaded:", output_file)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
```

4. Writes a manifest row (shard id, git-style code hash, seed list, config hash, rows written) to `manifests/shard{SHARD_ID:02d}.json` so completeness can be checked mechanically after download.
5. Resume logic: on start, reads any partial CSV in the working dir and skips completed seeds.

Workflow: local pilot (1 seed, smallest grid) -> measured extrapolation -> offload decision -> upload notebooks (up to 40 across accounts) -> collect CSVs -> completeness check against manifests -> aggregate in `scripts/aggregate_results.py`. Expected Colab need: Phase 2 probably none (maybe 2-4 notebooks for k_Y = 8 LP baselines); Phase 3 likely 4-8 for OT-heavy sweeps; Phase 4 as tabled (36 + reserve). Never exceed 40 outstanding.

---

## 9. Risk register

| Risk | Probability | Damage | Earliest detector | Prevention | Recovery | Terminal? | Owner package |
|---|---|---|---|---|---|---|---|
| Direct prior-art hit (CbD or quantum community publishes causal deployment) | Low-med | High | WP 1.1 re-search; monthly arXiv alert `contextuality AND causal` | Lead with dictionary + continuous layer | Reposition as empirical benchmark paper | Possibly | WP 1.1 |
| Weak-baseline illusion (untuned LP makes witnesses look good) | Med | High | WP 2.3 equal-budget protocol; external sanity run of `inflation` defaults | Equalized budgets, timeouts logged | Re-run contested cells tuned | No | WP 2.3 |
| Population-level redundancy perceived as fatal by referees | Med | Med | WP 1.3 redundancy table | Honest framing from the abstract (statistical, not informational) | Lean harder on continuous layer | No | WP 1.3 |
| Nonidentification of the target miscommunicated | Med | High | WP 1.2 memo review | Shah-Peters anchoring, named alternative classes | Scope statements | No | WP 1.2 |
| Crackle/heavy-tail failure of continuous witness | Med-high | High | WP 3.2 | Trimming predeclared; tail class named | Scope to sub-Gaussian | Yes if unfixable | WP 3.2 |
| Data hunger (witness needs n > 10^6) | Med | High | WP 3.3 early sweep at n = 500 | Effect-size-first design | None if triggered | Yes | WP 3.3 |
| Tuning leakage (bandwidth chosen on test data) | Med | Med | Protocol review before WP 3.3 | Frozen h-grids; selection on calibration split only | Redo affected cells | No | WP 3.1-3.3 |
| Colab logistics (account juggling, lost shards) | Med | Low | Manifest completeness check | Manifests + resume logic + 40-cap | Re-run lost shards | No | Section 8 |
| Decorative-theory drift | Med | Med | Phase 5 stop rules | Tags + stop rules above | Cut theorems | No | Phase 5 |
| Diffuse paper story (dictionary vs statistics vs continuity) | Med | Med | Outline review at Gate D | One-sentence contribution test | Restructure around surviving arm | No | Phase 5 |
| Scooping (Sapienza or inflation community moves to sampling theory) | Low-med | Med | Monthly alerts | Speed via cheap decisive phases | Emphasize continuous layer | No | all |

---

## 10. Reproducibility and artifact map

```text
Contextual_Confounding/            (new project root; create under ~/Stuff/Research/Topological_Confounding_Detection/ or standalone)
  src/
    dgps.py                        seeded DGP library (all Phase 1-4 structures)
    witnesses.py                   discrete witnesses + estimators
    continuous_witness.py          K1/K2 constructions
    baselines/
      inflation_lp.py              wrapper around github.com/ecboghiu/inflation (pin commit)
      nested_markov.py             Ananke score-test wrapper (pin commit)
      ineq_tests.py                Pearl / Bonet / Kedagni-Mourifie tests
      ci_tests.py                  HSIC/KCI, knockpy CRT
    enumeration.py                 WP 1.3/1.4 drivers
    calibration.py                 bootstrap / subsampling / CRT / multiplier
  configs/seeds.json               master seed registry
  configs/*.yaml                   frozen experiment configs (committed before runs)
  results/phase1_enumeration/ ... results/phase4_grid/   (CSV only, schemas in schemas.md)
  schemas.md                       column dictionary per CSV
  figures/                         every figure regenerable from CSVs by scripts/make_figures.py
  colab/templates/, colab/shards/  notebooks + downloaded outputs
  manifests/                       shard manifests
  memos/gate_[A-E]_memo.md         gate decisions
  requirements.txt + environment.yml
```

Policies: configs committed before the runs they describe; results directories append-only; failures logged not deleted; environment locked (pip freeze committed); every table/figure traceable to (config hash, seed list, code hash). Raw-vs-processed boundary: `results/**` is processed only; raw replication-level dumps live in `results/raw/` and are never edited.

---

## 11. Immediate actions (stop at Gate A)

1. WP 1.1: run the four additional prior-art query strings; inspect Navascues-Wolfe completeness statement and Balke-Pearl LP; update `memos/prior_art_ledger.csv`. (Half day)
2. WP 1.2: draft `memos/formalization_memo.md` with the F1/F2-fixed objects and two hand-worked micro-examples; sub-agent review pass. (1-2 days)
3. WP 1.3: implement `src/enumeration.py` + exact LP ground truth + population witnesses; run the 10^6-point scan; produce the four CSVs. (About 1 week)
4. WP 1.4: structure atlas on reduced density. (2-3 days)
5. WP 1.5: write `memos/gate_A_memo.md`; decide GO / PIVOT / INCREMENTAL-ONLY / KILL together with the user.

No theory writing, no Colab setup, and no Phase 2+ work before Gate A clears.

---

## 12. References

Core foundations:

- Abramsky, Brandenburger (2011). The sheaf-theoretic structure of non-locality and contextuality. New J. Phys. 13, 113036. https://arxiv.org/abs/1102.0264 , https://doi.org/10.1088/1367-2630/13/11/113036
- Wolfe, Spekkens, Fritz (2019). The inflation technique for causal inference with latent variables. J. Causal Inference 7(2). https://arxiv.org/abs/1609.00672 , https://doi.org/10.1515/jci-2017-0020
- Navascues, Wolfe (2019). The inflation technique solves completely the classical inference problem. J. Causal Inference 8(1). https://arxiv.org/abs/1707.06476 , https://doi.org/10.1515/jci-2018-0008
- Pearl (2009). Causality (2nd ed.), Ch. 8. Cambridge Univ. Press. https://doi.org/10.1017/CBO9780511803161
- Shah, Peters (2020). The hardness of conditional independence testing and the generalised covariance measure. Ann. Statist. 48(3). https://arxiv.org/abs/1804.07203 , https://doi.org/10.1214/19-AOS1851

Baselines and comparators:

- Balke, Pearl (1997). Bounds on treatment effects from studies with imperfect compliance. JASA 92(439). <!-- [verify] DOI digits: 10.1080/01621459.1997.10474074 -->
- Richardson, Evans, Shpitser, Robins. Nested Markov models for AD MGs. Ann. Statist. <!-- [verify] exact year/volume/pages before bib entry -->
- Kedagni, Mourifie. Generalized instrumental inequalities: testing the IV independence assumption. Biometrika. https://arxiv.org/abs/1809.05660 <!-- [verify] journal DOI before bib entry -->
- Bonet. Instrumentality, reversibility, and sufficient conditions for context-specific independence? / Bonet's inequality constraints. <!-- [verify] exact Bonet reference for the inequality constraints before citing -->
- Chaves et al. Inferring latent structures via information inequalities (UAI 2014). https://arxiv.org/abs/1307.3566 <!-- [verify] venue/year before bib entry -->

Adjacent contextuality lines (to cite and differentiate):

- Dzhafarov, Kujala and the Contextuality-by-Default literature; tutorial: de Barros, Kujala, Oas (2016). Negative probabilities and contextuality. https://arxiv.org/abs/1511.02823 ; Cervantes, Dzhafarov (2019). True contextuality in a psychophysical experiment (bootstrap reliability practice). https://arxiv.org/abs/1812.00105
- Gogioso, Pinzani trilogy. The Combinatorics/Topology/Geometry of Causality. https://arxiv.org/abs/2206.08911 , https://arxiv.org/abs/2303.07148 , https://arxiv.org/abs/2303.09017
- Sargsyan (2026). A cubical formalisation of topos causal models: intervention, forcing, and a contextuality obstruction. https://arxiv.org/abs/2607.15629
- Vallee, Markham (2025). Formalizing contextuality in sequential scenarios. https://arxiv.org/abs/2509.14125
- Pozas-Kerstjens, Gisin, Renou (2023). Proofs of network quantum nonlocality in continuous families of distributions (continuous-parameter inflation usage). PRL 130, 090201. https://arxiv.org/abs/2203.16543 , https://doi.org/10.1103/PhysRevLett.130.090201

Software:

- inflation: https://github.com/ecboghiu/inflation ; Ananke: https://gitlab.com/causal/ananke ; knockpy: https://github.com/amspector100/knockpy ; causal-learn: https://github.com/py-why/causal-learn ; POT: https://pythonot.github.io ; lingam (RCD baselines): https://github.com/cdt15/lingam

Internal:

- B2 audit (impossibility discipline, crackle caveat, monodromy ancestry): `Topological_Confounding_Detection/B2_topological_confounding_detection.md`
