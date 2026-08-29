"""Generate Colab shard notebooks for the Phase 4 continuous null arm
(p4cnull) — the ONLY block that per D-P4.5 may need offloading (24 groups
= n {8000, 20000} x d {2,3,5} x noise {gauss,t3} x kinds, B=199, ~50-60
min/group measured).

Local overnight run may finish these first; run this generator ONLY for
groups whose raw CSV is still missing in results/raw/phase4/ in the
morning. Notebooks are pinned to the CURRENT HEAD — commit and push
BEFORE uploading (the clone cell checks out GIT_SHA).

Usage: python3 scripts/make_colab_shards_p4.py [--shards 12]
Writes colab/shards/ccx_p4cnull_shardXX.ipynb (2 groups per shard at 12).
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SHARD_DIR = ROOT / "colab" / "shards"
GIT_URL = "https://github.com/hugogobato/ccx-contextual-confounding.git"

DRIVER = """
# ---- p4cnull driver (sharded, sequential; D-P4.8 fixed code) ----
import os, json, time
import numpy as np, pandas as pd
from run_p4_continuous import process_null_group, make_null_groups

p4 = json.loads(open("configs/phase4.json").read())
all_groups = make_null_groups(p4, pilot=False)
todo = [all_groups[i] for i in CONFIG["group_ids"]]
print(f"shard {SHARD_ID}: {len(todo)} groups")

rows_all, done_keys = [], set()
if os.path.exists(OUT):
    try:
        prev = pd.read_csv(OUT)
        if len(prev):
            done_keys = set(zip(prev["n"], prev["d"], prev["noise"],
                                prev["kind"], prev["seed"],
                                prev["method"], prev["trim"]))
            rows_all = prev.to_dict("records")
            print(f"resume: {len(rows_all)} rows")
    except Exception as e:
        print("resume failed", e)

for gi, g in enumerate(todo):
    print(f"[{gi+1}/{len(todo)}] n={g['n']} d={g['d']} {g['noise']} "
          f"{g['kind']} B={g['B']} seeds={len(g['seeds'])}", flush=True)
    t0 = time.time()
    try:
        rows = process_null_group(g)
    except Exception as e:
        import traceback; traceback.print_exc(); continue
    rows = [r for r in rows
            if (r["n"], r["d"], r["noise"], r["kind"], r["seed"],
                r["method"], r["trim"]) not in done_keys]
    rows_all.extend(rows)
    if rows_all:
        pd.DataFrame(rows_all).to_csv(OUT, index=False)
    print(f"  {len(rows)} new rows, {time.time()-t0:.0f}s", flush=True)

manifest = {"tag": TAG, "shard_id": SHARD_ID, "code_hash": CODE_HASH,
            "groups": [int(i) for i in CONFIG["group_ids"]],
            "rows": len(rows_all), "git_sha": GIT_SHA}
mpath = f"/content/ccx_{TAG}_manifest_shard{SHARD_ID:02d}.json"
with open(mpath, "w") as fh:
    json.dump(manifest, fh, indent=2)
print("MANIFEST", json.dumps(manifest))
try:
    from google.colab import files
    files.download(OUT); files.download(mpath)
except Exception as e:
    print("(download skipped)", e)
"""


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        raise SystemExit("git sha unavailable; commit first")


def nb_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


CLONE = """# ---- setup: clone pinned repo + deps ----
import os, sys, subprocess
REPO_DIR = "/tmp/ccx"
GIT_URL = "{url}"
GIT_SHA = "{sha}"
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "scipy==1.17.1", "pandas==3.0.1"])
except Exception as e:
    print("pip warning:", e)
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    tok = None
    try:
        from google.colab import userdata
        tok = userdata.get("CCX_GH_TOKEN")
    except Exception:
        tok = os.environ.get("CCX_GH_TOKEN")
    url = GIT_URL if not tok else GIT_URL.replace("https://",
                                                  f"https://{{tok}}@")
    subprocess.check_call(["git", "clone", url, REPO_DIR])
subprocess.check_call(["git", "-C", REPO_DIR, "checkout", GIT_SHA, "-q"])
sys.path.insert(0, REPO_DIR + "/src")
os.chdir(REPO_DIR)
import numpy, scipy, pandas
print("checked out", GIT_SHA[:7], "| numpy", numpy.__version__)
CODE_HASH = GIT_SHA[:16]
""".replace("{url}", GIT_URL)


def main():
    n_shards = int(sys.argv[sys.argv.index("--shards") + 1]) \
        if "--shards" in sys.argv else 12
    sha = git_sha()
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    from run_p4_continuous import make_null_groups
    p4 = json.loads((ROOT / "configs" / "phase4.json").read_text())
    groups = make_null_groups(p4, pilot=False)
    # only groups missing locally
    todo = []
    for i, g in enumerate(groups):
        f = (ROOT / "results" / "raw" / "phase4" /
             f"p4cnull_n{g['n']}_d{g['d']}_{g['noise']}_{g['kind']}.csv")
        if not f.exists():
            todo.append(i)
    print(f"{len(todo)} of {len(groups)} groups missing -> {n_shards} shards")
    if not todo:
        print("nothing to offload; all p4cnull raw present locally")
        return
    per = (len(todo) + n_shards - 1) // n_shards
    cfg_cell = ("import os, json\nTAG='p4cnull'\nSHARD_ID=%d\n"
                "OUT='/content/ccx_p4cnull_shard%02d.csv'\n"
                "CONFIG_JSON=%r\nCONFIG=json.loads(CONFIG_JSON)\n")
    for s in range(n_shards):
        ids = todo[s * per:(s + 1) * per]
        if not ids:
            continue
        cfg = {"group_ids": ids}
        cells = [
            (cfg_cell % (s, s, json.dumps(cfg))).splitlines(keepends=True),
            CLONE.splitlines(keepends=True),
            DRIVER.splitlines(keepends=True),
        ]
        nb = {"nbformat": 4, "nbformat_minor": 5,
              "metadata": {"accelerator": "None",
                           "colab": {"provenance": []}},
              "cells": [nb_cell(c) for c in cells]}
        p = SHARD_DIR / f"ccx_p4cnull_shard{s:02d}.ipynb"
        p.write_text(json.dumps(nb))
        print("wrote", p, "groups", ids)


if __name__ == "__main__":
    main()
