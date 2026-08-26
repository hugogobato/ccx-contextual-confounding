"""Generate Colab shard notebooks for missing results.

Missing (deleted by rmtree): Phase 1 enumeration (1.12M), WP2.2 calibration,
WP2.3 power, WP2.4 pain map. Uses thin-clone style pinned to current HEAD
so notebooks stay small and deterministic; no inline source duplication.
Existing Phase3 shards (34) are untouched.

Outputs to colab/shards/:
  ccx_phase1_shard00..09  (10 shards, 12 jobs each, 120 total)
  ccx_wp22_shard00..17   (18 shards, 10 groups each, 180 total)
  ccx_wp23_shard00..08   (9 shards, 48 groups each, 432 total)
  ccx_wp24_shard00       (1 shard, full pain map)

Each notebook: cell1 CONFIG_JSON + SHARD_ID, cell2 clone+install, cell3 driver
with resume, incremental appends, manifest, download fallback (Plan Sec 8).
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_wp22_calibration import make_groups as wp22_groups, load_cfg as load_cfg_phase2
from run_wp23_power import make_groups as wp23_groups

SHARD_DIR = ROOT / "colab" / "shards"
GIT_URL = "https://github.com/hugogobato/ccx-contextual-confounding.git"

def git_sha():
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        return out
    except Exception:
        return "dfd0468"

GIT_SHA = git_sha()

# ---- notebook helpers ----
def nb_cell(source_lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }

def write_nb(path, cells):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"accelerator": "None", "colab": {"provenance": []}},
        "cells": [nb_cell(c) for c in cells],
    }
    Path(path).write_text(json.dumps(nb))
    print(f"wrote {path}")

def cell1_config(tag, shard_id, out_name, config):
    cfg_json = json.dumps(config)
    lines = [
        f"# CCX {tag.upper()} shard {shard_id:02d} — CONFIG (Plan Sec 8)\n",
        "import os, json\n",
        f"TAG='{tag}'\n",
        f"SHARD_ID={shard_id}\n",
        f"OUT='/content/{out_name}'\n",
        f"CONFIG_JSON={cfg_json!r}\n",
        "CONFIG=json.loads(CONFIG_JSON)\n",
        f"print(TAG, 'shard', SHARD_ID, 'groups', len(CONFIG.get('groups', CONFIG.get('jobs', []))))\n",
    ]
    return lines

CLONE_SETUP = f"""# ---- setup: clone pinned repo + deps (thin) ----
import os, sys, subprocess, pathlib, json, time, hashlib

REPO_DIR = "/tmp/ccx"
GIT_URL = "{GIT_URL}"
GIT_SHA = "{GIT_SHA}"

# pip deps (numpy/scipy/pandas/matplotlib already on Colab, ensure versions)
# keep install light; inflation not needed for these jobs but harmless
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy==1.17.1", "pandas==3.0.1"])
except Exception as e:
    print("pip install warning:", e)

# clone or reuse
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    print(f"cloning {{GIT_URL}} @ {{GIT_SHA[:7]}} -> {{REPO_DIR}}")
    # try anonymous clone first (works if repo public); if private, try token from Colab secrets
    tok = None
    try:
        from google.colab import userdata
        tok = userdata.get("CCX_GH_TOKEN")
    except Exception:
        tok = os.environ.get("CCX_GH_TOKEN")
    url = GIT_URL
    if tok:
        # inject token: https://<token>@github.com/...
        url = GIT_URL.replace("https://", f"https://{{tok}}@")
        print("using GH token from secrets/env")
    subprocess.check_call(["git", "clone", url, REPO_DIR])
else:
    print("repo already cloned, fetching")
    subprocess.check_call(["git", "-C", REPO_DIR, "fetch", "--all", "-q"])

subprocess.check_call(["git", "-C", REPO_DIR, "checkout", GIT_SHA, "-q"])
# ensure src on path and ROOT env for imports
if REPO_DIR + "/src" not in sys.path:
    sys.path.insert(0, REPO_DIR + "/src")
os.chdir(REPO_DIR)
print("checked out", subprocess.check_output(["git","-C",REPO_DIR,"rev-parse","--short","HEAD"], text=True).strip())
# verify
import numpy, scipy, pandas
print("numpy", numpy.__version__, "scipy", scipy.__version__, "pandas", pandas.__version__)
CODE_HASH = GIT_SHA[:16]
"""

def cell2_clone():
    return [l + "\n" for l in CLONE_SETUP.splitlines(keepends=False)]

# ---- drivers ----

DRIVER_PHASE1 = """
# ---- Phase 1 enumeration driver (sharded) ----
import numpy as np, pandas as pd
from pathlib import Path
from models import build_iv_A, iv_vertices, discover_facets
from witnesses import slack_and_feasible, cf1_soft, degree_signed, kl_em_batch, iv_cf1_hard, maximal_support_lp, iv_inflation2_feasible
from dgps import uniform_conditionals, boundary_batch, sparse_conditionals
import json as _json

# load facets (generate if missing)
FACETS_CACHE = Path("results/phase1_enumeration/facets_iv.npz")
if FACETS_CACHE.exists():
    z = np.load(FACETS_CACHE)
    H_fac, b_fac = z["H"], z["b"]
else:
    print("facets cache missing, generating via discover_facets...")
    H_fac, b_fac = discover_facets(build_iv_A(), seed=0)
    FACETS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FACETS_CACHE, H=H_fac, b=b_fac)
    print("facets generated:", H_fac.shape)

A16 = build_iv_A()
Af = A16.astype(float)
TOL_BORDER_LO, TOL_BORDER_HI = 1e-9, 1e-7

def process_batch_inline(E, source, batch_label, seed):
    n = len(E)
    fac_ok = np.all(H_fac.astype(float) @ E.T <= b_fac[:, None].astype(float) + 1e-9, axis=0)
    kl_vals, _ = kl_em_batch(Af, E)
    rows=[]
    for i in range(n):
        e=E[i]
        slack, feas = slack_and_feasible(Af, e)
        border = TOL_BORDER_LO < slack < TOL_BORDER_HI
        row={"instance_id": f"{source}_{batch_label}_{i}", "source": source, "batch": batch_label, "seed": seed,
             **{f"e_{j}": float(e[j]) for j in range(8)},
             "lp_feasible": bool(feas), "facets_ok": bool(fac_ok[i]), "agree_c1a": bool(feas==bool(fac_ok[i])),
             "slack_l1": float(slack), "borderline_band": bool(border),
             "cf1_soft": cf1_soft(Af, e), "degree_signed": degree_signed(Af, e),
             "kl_contextuality": float(kl_vals[i]), "inflation2_feasible": np.nan}
        if not feas and not border:
            row["cf1_hard"]=iv_cf1_hard(e, A16)
            m0=maximal_support_lp(A16[0:4], e[0:4]>0)
            m1=maximal_support_lp(A16[4:8], e[4:8]>0)
            row["maximally_contextual"]=bool(max(m0,m1)<=1e-9)
        else:
            row["cf1_hard"]=np.nan
            row["maximally_contextual"]=False
        rows.append(row)
    return rows

# resume: check which batches already in OUT
rows_all=[]
done_batches=set()
if os.path.exists(OUT):
    try:
        prev=pd.read_csv(OUT)
        if "batch" in prev.columns and "source" in prev.columns:
            done_batches=set(zip(prev["source"], prev["batch"].astype(int)))
        rows_all=prev.to_dict("records")
        print(f"resume: {len(done_batches)} batches, {len(rows_all)} rows already in OUT")
    except Exception as e:
        print("resume read failed", e)

jobs=CONFIG["jobs"]
todo=[j for j in jobs if (j["source"], j["batch"]) not in done_batches]
print(f"jobs total {len(jobs)} todo {len(todo)}")

for idx, job in enumerate(todo):
    src=job["source"]; batch=job["batch"]; seed=job["seed"]; n=job["n"]
    print(f"[{idx+1}/{len(todo)}] {src} batch={batch} n={n} seed={seed}", flush=True)
    t0=time.time()
    if src=="uniform":
        rng=np.random.default_rng(seed)
        E=uniform_conditionals(rng, n)
    elif src=="sparse":
        rng=np.random.default_rng(seed)
        # sparse: use dgps sparse_conditionals directly
        E=sparse_conditionals(rng, n)
    else:  # boundary
        E=boundary_batch(A16, H_fac, b_fac, n, seed)
    rows=process_batch_inline(E, src, batch, seed)
    rows_all.extend(rows)
    # incremental save every job (small jobs, cheap)
    pd.DataFrame(rows_all).to_csv(OUT, index=False)
    print(f"  -> {len(rows)} rows, {time.time()-t0:.1f}s, total {len(rows_all)}", flush=True)

# final manifest
manifest={"tag": TAG, "shard_id": SHARD_ID, "code_hash": CODE_HASH, "jobs": len(jobs), "rows": len(rows_all), "git_sha": GIT_SHA}
mpath=f"/content/ccx_{TAG}_manifest_shard{SHARD_ID:02d}.json"
with open(mpath,"w") as fh: json.dump(manifest,fh,indent=2)
print("MANIFEST", json.dumps(manifest))
try:
    from google.colab import files
    files.download(OUT); files.download(mpath)
    print("downloaded", OUT)
except Exception as e:
    print("(download skipped)", e)
"""

DRIVER_WP22 = """
# ---- WP2.2 calibration driver (sharded, sequential) ----
import os, json, time, hashlib
import numpy as np, pandas as pd
from pathlib import Path
# imports from cloned repo src
from run_wp22_calibration import process_group, load_cfg
from models import build_iv_A_general  # needed for facet/battery caches side effects

# ensure binary facets exist (WP2.2's get_binary_facets loads from file; generate if missing)
from pathlib import Path as _P
_facets = _P("results/phase1_enumeration/facets_iv.npz")
if not _facets.exists():
    print("facets missing, generating...")
    from models import build_iv_A, discover_facets
    Hf, bf = discover_facets(build_iv_A(), seed=0)
    _facets.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_facets, H=Hf, b=bf)
    print("facets generated", Hf.shape)

cfg=load_cfg()
groups=CONFIG["groups"]
print(f"groups in shard: {len(groups)}")

# OUT is a single combined csv per shard (incremental)
rows_all=[]
done_keys=set()
if os.path.exists(OUT):
    try:
        prev=pd.read_csv(OUT)
        if len(prev):
            # key: cell, kind, n, seed, arm, stat, engine
            done_keys=set((r["cell"], r["kind"], int(r["n"]), int(r["seed"]), r["arm"], r["stat"], r["engine"]) for _,r in prev.iterrows())
            rows_all=prev.to_dict("records")
            print(f"resume: {len(done_keys)} row-keys, {len(rows_all)} rows")
    except Exception as e:
        print("resume failed", e)

for gi, g in enumerate(groups):
    # g has cell/kind/n/seeds/boot_seeds/cfg stripped; reattach cfg
    g["cfg"]=cfg
    # need tuple cell
    g["cell"]=tuple(g["cell"])
    print(f"[{gi+1}/{len(groups)}] {g['cell']} {g['kind']} n={g['n']} seeds={len(g['seeds'])} boot={len(g['boot_seeds'])}", flush=True)
    t0=time.time()
    try:
        rows=process_group(g)
    except Exception as e:
        print(f"  ERROR group {g['cell']} {g['kind']} n={g['n']}: {e}")
        import traceback; traceback.print_exc()
        continue
    # filter to unsaved rows (if resume, skip already done keys)
    if done_keys:
        new_rows=[r for r in rows if (r["cell"], r["kind"], int(r["n"]), int(r["seed"]), r["arm"], r["stat"], r["engine"]) not in done_keys]
        print(f"  produced {len(rows)} rows, {len(new_rows)} new after dedup")
        rows=new_rows
    else:
        print(f"  produced {len(rows)} rows")
    rows_all.extend(rows)
    # incremental save every group (WP2.2 groups are heavy)
    if rows_all:
        pd.DataFrame(rows_all).to_csv(OUT, index=False)
        # update done_keys
        for r in rows:
            done_keys.add((r["cell"], r["kind"], int(r["n"]), int(r["seed"]), r["arm"], r["stat"], r["engine"]))
    print(f"  group time {time.time()-t0:.1f}s total rows {len(rows_all)}", flush=True)

manifest={"tag": TAG, "shard_id": SHARD_ID, "code_hash": CODE_HASH, "groups": len(groups), "rows": len(rows_all), "git_sha": GIT_SHA}
mpath=f"/content/ccx_{TAG}_manifest_shard{SHARD_ID:02d}.json"
with open(mpath,"w") as fh: json.dump(manifest,fh,indent=2)
print("MANIFEST", json.dumps(manifest))
try:
    from google.colab import files
    files.download(OUT); files.download(mpath)
    print("downloaded", OUT)
except Exception as e:
    print("(download skipped)", e)
"""

DRIVER_WP23 = """
# ---- WP2.3 power driver (sharded, sequential) ----
import os, json, time
import numpy as np, pandas as pd
from run_wp23_power import process_group, load_cfg
# ensure facets for 2-2-2 cell
from pathlib import Path as _P
_facets = _P("results/phase1_enumeration/facets_iv.npz")
if not _facets.exists():
    print("facets missing, generating...")
    from models import build_iv_A, discover_facets
    Hf, bf = discover_facets(build_iv_A(), seed=0)
    _facets.parent.mkdir(parents=True, exist_ok=True)
    np.savez(_facets, H=Hf, b=bf)
    print("facets generated", Hf.shape)
cfg=load_cfg()
groups=CONFIG["groups"]
print(f"groups in shard: {len(groups)}")
rows_all=[]
done_keys=set()
if os.path.exists(OUT):
    try:
        prev=pd.read_csv(OUT)
        if len(prev):
            done_keys=set((r["cell"], r["family"], float(r["rho"]), int(r["n"]), int(r["seed"]), r["stat"], r["engine"]) for _,r in prev.iterrows())
            rows_all=prev.to_dict("records")
            print(f"resume {len(done_keys)} row-keys {len(rows_all)} rows")
    except Exception as e:
        print("resume failed", e)
for gi, g in enumerate(groups):
    g["cfg"]=cfg
    g["cell"]=tuple(g["cell"])
    print(f"[{gi+1}/{len(groups)}] {g['cell']} {g['family']} rho={g['rho']} n={g['n']} seeds={len(g['seeds'])}", flush=True)
    t0=time.time()
    try:
        rows=process_group(g)
    except Exception as e:
        print(f"  ERROR {g}: {e}")
        import traceback; traceback.print_exc()
        continue
    if done_keys:
        new_rows=[r for r in rows if (r["cell"], r["family"], float(r["rho"]), int(r["n"]), int(r["seed"]), r["stat"], r["engine"]) not in done_keys]
        print(f"  {len(rows)} -> {len(new_rows)} new")
        rows=new_rows
    rows_all.extend(rows)
    if rows_all:
        pd.DataFrame(rows_all).to_csv(OUT, index=False)
        for r in rows:
            done_keys.add((r["cell"], r["family"], float(r["rho"]), int(r["n"]), int(r["seed"]), r["stat"], r["engine"]))
    print(f"  {time.time()-t0:.1f}s total {len(rows_all)}", flush=True)

manifest={"tag": TAG, "shard_id": SHARD_ID, "code_hash": CODE_HASH, "groups": len(groups), "rows": len(rows_all), "git_sha": GIT_SHA}
mpath=f"/content/ccx_{TAG}_manifest_shard{SHARD_ID:02d}.json"
with open(mpath,"w") as fh: json.dump(manifest,fh,indent=2)
print("MANIFEST", json.dumps(manifest))
try:
    from google.colab import files
    files.download(OUT); files.download(mpath)
    print("downloaded", OUT)
except Exception as e:
    print("(dl skip)", e)
"""

DRIVER_WP24 = """
# ---- WP2.4 scaling pain map driver (single shard) ----
import os, json, time, pandas as pd
from run_wp24_scaling import bench_cell, get_battery_cached, build_iv_A_general
from pathlib import Path
import numpy as np
# bench_cell already does per-cell loops; we just call main-like logic but write to OUT

# reuse bench logic but ensure OUT path is shard file
from run_wp24_scaling import ROOT as WP24_ROOT
import run_wp24_scaling as wp24mod

# monkey-patch RES to point to /content for this shard? Instead run inline bench and write to OUT
# Copy bench loop from run_wp24_scaling.main but write to OUT

import json as _json
from pathlib import Path as _P
ROOT_SHARD = Path("/tmp/ccx")
# run bench
cfg=json.loads((ROOT_SHARD / "configs" / "seeds.json").read_text())["phase2"]
rows=[]
for kz,kx,ky in [tuple(c) for c in cfg["alphabet_cells"]]:
    for n in (500,2000,8000):
        t0=time.time()
        rows+=bench_cell(kz,kx,ky,n)
        print(f"[wp24] ({kz},{kx},{ky}) n={n}: {time.time()-t0:.1f}s", flush=True)
df=pd.DataFrame(rows)
# sweep
sweep=[]
for (kx,ky) in [(2,2),(2,5),(3,5),(2,8)]:
    for kz in (2,3,4):
        A=build_iv_A_general(kz,kx,ky)
        Mint=A.astype(int)
        K,Q=A.shape
        for r in range(3):
            rng=np.random.default_rng(77000+r)
            cond=np.concatenate([rng.dirichlet(np.ones(K//kz)) for _ in range(kz)])
            from run_wp24_scaling import timed
            from witness_estimators import cf1_plugin_stat, slack_plugin_stat
            dt1,m1,_=timed(cf1_plugin_stat, Mint, np.round(cond*4000), kz)
            dt2,m2,_=timed(slack_plugin_stat, Mint, np.round(cond*4000), kz)
            sweep.append({"kz":kz,"kx":kx,"ky":ky,"K":K,"Q":Q,"method":"cf1_lp","seconds":dt1,"peak_bytes":m1})
            sweep.append({"kz":kz,"kx":kx,"ky":ky,"K":K,"Q":Q,"method":"slack_lp","seconds":dt2,"peak_bytes":m2})
if sweep:
    df=pd.concat([df, pd.DataFrame(sweep)], ignore_index=True)
df.to_csv(OUT, index=False)
print(f"wrote {len(df)} rows -> {OUT}")

manifest={"tag": TAG, "shard_id": SHARD_ID, "code_hash": CODE_HASH, "rows": len(df), "git_sha": GIT_SHA}
mpath=f"/content/ccx_{TAG}_manifest_shard{SHARD_ID:02d}.json"
with open(mpath,"w") as fh: json.dump(manifest,fh,indent=2)
print("MANIFEST", json.dumps(manifest))
try:
    from google.colab import files
    files.download(OUT); files.download(mpath)
except Exception as e:
    print("(dl skip)", e)
"""

def main():
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"GIT_SHA {GIT_SHA[:7]} {GIT_SHA}")

    # ---- Phase 1: 120 jobs -> 10 shards x 12 jobs ----
    seeds = json.loads((ROOT / "configs" / "seeds.json").read_text())
    u = seeds["phase1_uniform"]
    bd = seeds["phase1_boundary"]
    sp = seeds.get("phase1_sparse", {"n_points": 20000, "seed": 555000})

    jobs = []
    # uniform 100 batches
    for b in range(u["seed_start"], u["seed_start"] + u["n_batches"]):
        jobs.append({"source": "uniform", "batch": b, "n": u["batch_size"], "seed": b})
    # boundary 100k -> 10 chunks of 10k
    bchunk = max(u["batch_size"], 2000)
    remaining = bd["n_points"]
    cid = 0
    while remaining > 0:
        take = min(bchunk, remaining)
        jobs.append({"source": "boundary", "batch": cid, "n": take, "seed": bd["seed"] + cid})
        remaining -= take
        cid += 1
    # sparse 20k -> 10 chunks of 2k
    remaining = sp["n_points"]
    cid = 0
    while remaining > 0:
        take = min(2000, remaining)
        jobs.append({"source": "sparse", "batch": cid, "n": take, "seed": sp["seed"] + cid})
        remaining -= take
        cid += 1

    n_phase1_shards = 10
    per = (len(jobs) + n_phase1_shards - 1) // n_phase1_shards
    for s in range(n_phase1_shards):
        chunk = jobs[s*per:(s+1)*per]
        config = {"jobs": chunk}
        c1 = cell1_config("phase1", s, f"ccx_phase1_shard{s:02d}.csv", config)
        c2 = cell2_clone()
        c3 = [l+"\n" for l in DRIVER_PHASE1.splitlines()]
        write_nb(SHARD_DIR / f"ccx_phase1_shard{s:02d}.ipynb", [c1, c2, c3])

    # ---- WP2.2: 180 groups -> 18 shards x 10 groups ----
    cfg2 = load_cfg_phase2()
    cells = [tuple(c) for c in cfg2["alphabet_cells"]]
    kinds = cfg2["wp22_null_kinds"]
    groups22 = wp22_groups(cfg2, cells, kinds, pilot=False)
    # strip cfg to keep CONFIG small (reloaded in notebook)
    for g in groups22:
        g.pop("cfg", None)
        g["cell"] = list(g["cell"])
    n22 = 18
    per = (len(groups22) + n22 - 1) // n22
    for s in range(n22):
        chunk = groups22[s*per:(s+1)*per]
        config = {"groups": chunk}
        c1 = cell1_config("wp22", s, f"ccx_wp22_shard{s:02d}.csv", config)
        c2 = cell2_clone()
        c3 = [l+"\n" for l in DRIVER_WP22.splitlines()]
        write_nb(SHARD_DIR / f"ccx_wp22_shard{s:02d}.ipynb", [c1, c2, c3])

    # ---- WP2.3: 432 groups -> 9 shards x 48 groups ----
    groups23 = wp23_groups(cfg2, ["mixture","mechanistic"], pilot=False)
    for g in groups23:
        g.pop("cfg", None)
        g["cell"] = list(g["cell"])
    n23 = 9
    per = (len(groups23) + n23 - 1) // n23
    for s in range(n23):
        chunk = groups23[s*per:(s+1)*per]
        config = {"groups": chunk}
        c1 = cell1_config("wp23", s, f"ccx_wp23_shard{s:02d}.csv", config)
        c2 = cell2_clone()
        c3 = [l+"\n" for l in DRIVER_WP23.splitlines()]
        write_nb(SHARD_DIR / f"ccx_wp23_shard{s:02d}.ipynb", [c1, c2, c3])

    # ---- WP2.4: 1 shard ----
    config = {"note": "full pain map, single shard"}
    c1 = cell1_config("wp24", 0, "ccx_wp24_shard00.csv", config)
    c2 = cell2_clone()
    c3 = [l+"\n" for l in DRIVER_WP24.splitlines()]
    write_nb(SHARD_DIR / "ccx_wp24_shard00.ipynb", [c1, c2, c3])

    print(f"done: phase1 {n_phase1_shards} + wp22 {n22} + wp23 {n23} + wp24 1 -> {SHARD_DIR}")
    # also print existing phase3 counts for reference
    existing = sorted(SHARD_DIR.glob("ccx_wp32_*.ipynb")) + sorted(SHARD_DIR.glob("ccx_wp33_*.ipynb"))
    print(f"existing phase3 shards untouched: {len(existing)}")

if __name__ == "__main__":
    main()
