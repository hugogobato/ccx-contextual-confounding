# Colab shards — how to run missing jobs

Pinned code: `d2dd265` (all source, drivers, tests, configs). Deterministic: same seeds → identical numbers; spot-check against gate memos after re-run.

## Phase 4 (2026-08-29)

Per D-P4.5, Phase 4 was sized to run LOCALLY (discrete blocks measure
0.1-0.2 s/seed; continuous alt/adversarial blocks are minutes-to-an-hour
jobs). The only candidate for offload is `p4cnull` (24 groups: n in
{8000, 20000} x d {2,3,5} x noise {gauss,t3} x kinds, B=199, ~50-60
min/group), which is being attempted locally first.

IF any `p4cnull` groups are missing in the morning (check
`results/raw/phase4/overnight.log` and the p4cnull_*.csv files):

1. `python3 scripts/make_colab_shards_p4.py --shards 12` — generates
   notebooks ONLY for groups still missing, pinned to current HEAD.
   Commit and push BEFORE generating (the clone cell checks out GIT_SHA).
2. Upload `colab/shards/ccx_p4cnull_shardXX.ipynb` to Colab, Runtime →
   Run all. Each writes `ccx_p4cnull_shardXX.csv` +
   `ccx_p4cnull_manifest_shardXX.json` and auto-downloads.
3. Place downloads in `results/raw/phase4/` and re-run
   `python3 src/aggregate_phase4.py` (it picks up Colab shards
   automatically; keys are (n, d, noise, kind, seed, method, trim)).

## What's here (72 notebooks)

* **Already validated** (do not re-run unless needed):
  * `ccx_wp32_shard00..17` (18 × 3 groups) — WP3.2 null calibration
  * `ccx_wp33_shard00..15` (16 shards, 810 groups) — WP3.3 separation

* **New — missing results (deleted by rmtree), regenerate these:**
  * `ccx_phase1_shard00..09` (10 shards, 12 jobs each) — Phase 1 enumeration 1.12 M instances (~1–2 h total, ~8–12 min/shard)
  * `ccx_wp22_shard00..17` (18 shards, 10 groups each) — WP2.2 calibration suite (~10–14 h total, ~35–45 min/shard)
  * `ccx_wp23_shard00..08` (9 shards, 48 groups each) — WP2.3 power study (~40–60 min total, ~5–7 min/shard)
  * `ccx_wp24_shard00` (1 shard) — WP2.4 pain map (~10 min)

All shards: thin-clone style (pip + `git clone https://github.com/hugogobato/ccx-contextual-confounding.git @ d2dd265` + driver). Resume-safe (re-run skips done keys), incremental saves, manifest json, download fallback (Plan Sec 8).

## If repo is private

If `git clone` fails with 403, either:
* make the GitHub repo public, or
* add a fine-grained PAT (read-only, contents:read) as Colab secret `CCX_GH_TOKEN` (left panel → Secrets → `CCX_GH_TOKEN`), or set env `CCX_GH_TOKEN` before running.

The clone cell tries `userdata.get("CCX_GH_TOKEN")` automatically.

## Running

1. Open `colab/shards/ccx_<tag>_shardXX.ipynb` in Colab (you can run up to 16 accounts × 3 parallel = 48 concurrent).
2. **Runtime → Run all**. Each shard writes `/content/ccx_<tag>_shardXX.csv` and `/content/ccx_<tag>_manifest_shardXX.json` and auto-downloads them (also via `files.download` fallback).
3. If a shard hits a 12 h limit or disconnects, just **Run all again** — it resumes from `OUT`.

Suggested parallelization (fits 16 accounts):
* Phase1 (10 shards) → 10 accounts, ~12 min each
* WP2.4 (1) → 1 account, ~10 min (run first, quick)
* WP2.3 (9) → 9 accounts, ~6 min each
* WP2.2 (18) → 18 accounts in two waves (~40 min/shard) or 9 accounts × 2 waves

WP2.2 is the long pole; run it overnight.

## After download: aggregation (local)

Place downloads under the repo:

```bash
mkdir -p results/raw/phase1 results/raw/phase2 results/phase2_discrete results/phase1_enumeration
# Phase1: 10 files
mv ~/Downloads/ccx_phase1_shard*.csv results/raw/phase1/  # or keep as colab/shards downloads
# WP2.2: 18 files
mv ~/Downloads/ccx_wp22_shard*.csv results/raw/phase2/
# WP2.3: 9 files
mv ~/Downloads/ccx_wp23_shard*.csv results/raw/phase2/
# WP2.4: 1 file
mv ~/Downloads/ccx_wp24_shard*.csv results/phase2_discrete/lp_walltime_map.csv
# manifests (optional, for completeness check)
mv ~/Downloads/ccx_*_manifest_*.json results/raw/
```

Then aggregate:

```bash
# Phase1: concat shards + add vertices + inflation subsample → 4 final CSVs + facets
python scripts/aggregate_phase1_shards.py  # reads results/raw/phase1/ccx_phase1_*.csv or uniform_batch_*.csv
# → results/phase1_enumeration/{t1_dictionary.csv, hierarchy_placement.csv, strictness_scan.csv, witness_lp_redundancy.csv, facets_iv.npz}

# WP2.2: pool clean critical values, check pooled counts >=2400, write null_critical_values.csv + size tables
python scripts/aggregate_wp22_shards.py  # or: python src/aggregate_wp22.py  (handles both naming schemes)
# WP2.3: join with pooled CVs → power_curves.csv
python scripts/aggregate_wp23_shards.py  # or: src/aggregate_wp23.py
# WP2.4 already is lp_walltime_map.csv; optionally re-make figures
python src/make_figures.py
python src/make_figures_phase2.py
```

Manifest check (all shards must report `rows_written` >0):

```bash
python scripts/check_shard_manifests.py --dir results/raw --pattern "ccx_*_manifest_*.json"
python scripts/check_shard_manifests.py --dir results/raw/phase2 --pattern "ccx_*_manifest_*.json"
```

## Spot-check fidelity

After aggregation, compare key numbers to gate memos (should match exactly, deterministic):

* Phase1: `t1_dictionary.csv` 1,120,016 rows, `agree_c1a` 100% non-borderline, 267,020 contextual (23.84%), `witness_lp_redundancy.csv` cf1_soft rho_contextual = -1.0
* WP2.2: `null_critical_values.csv` pooled draws per (cell,n,stat,engine) ≥2400
* WP2.3: `power_curves.csv` join keys match pooled CVs; check a few power points at rho=0.5 n=2000

## Extras

* `scripts/make_colab_shards_all.py` — regenerates all 38 missing-job shards pinned to current HEAD (keeps Phase3 untouched). Re-run if seeds/config change.
* `colab/shards/manifest.json` — machine-readable index of all 72 shards.
