"""Generate THIN Colab shard notebooks for EVERYTHING needing regeneration:

  phase1    : 10 shards x 12 enumeration jobs
  phase1agg : 1 aggregation notebook (run after phase1 complete)
  wp22      : 18 shards x 10 calibration groups
  wp23      : 9 shards x ~48 power groups
  wp24      : 1 scaling pain map notebook
Total: 39 notebooks (<= 40 budget). Phase 3 suites (34) were issued earlier.

All notebooks clone the repo at pinned GIT_SHA and import src modules.
Usage: python3 scripts/make_colab_shards.py
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from run_wp32_calibration import make_groups as wp32_groups  # noqa: E402,F401
from run_wp22_calibration import make_groups as wp22_groups
from run_wp23_power import make_groups as wp23_groups

SHARD_DIR = ROOT / "colab" / "shards"
GIT_URL = "https://github.com/hugogobato/ccx-contextual-confounding.git"


def current_sha():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                       capture_output=True)
    if r.returncode == 0:
        return r.stdout.decode().strip()
    raise RuntimeError("determine GIT_SHA first")


def nb(cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "None",
                     "colab": {"provenance": []}},
        "cells": [{"cell_type": "code", "execution_count": None,
                   "metadata": {}, "outputs": [],
                   "source": c.splitlines(keepends=True)}
                  for c in cells],
    }


def cell_config(tag, sid, cfg):
    return ("# CCX %s shard %02d - config cell\n"
            "TAG = '%s'\nSHARD_ID = %d\n"
            "OUT = '/content/ccx_%s_shard%02d.csv'\n"
            "CONFIG_JSON = %r\n"
            "import json as _json\nCONFIG = _json.loads(CONFIG_JSON)\n"
            % (tag.upper(), sid, tag, sid, tag, sid,
               json.dumps(cfg)))


def cell_setup(git_sha):
    return (
        "# clone pinned repo and import modules\n"
        "import os, sys, shutil\n"
        "REPO = '/content/ccx-src'\n"
        "GIT_URL = '%s'\nGIT_SHA = '%s'\n" % (GIT_URL, git_sha)
        + "\n"
          "def _clone():\n"
          "    shutil.rmtree(REPO, ignore_errors=True)\n"
          "    r = os.system('git clone -q ' + GIT_URL + ' ' + REPO)\n"
          "    if r == 0:\n"
          "        return True\n"
          "    shutil.rmtree(REPO, ignore_errors=True)\n"
          "    tok = None\n"
          "    try:\n"
          "        from google.colab import userdata\n"
          "        tok = userdata.get('CCX_GH_PAT')\n"
          "    except Exception:\n"
          "        pass\n"
          "    if not tok:\n"
          "        raise RuntimeError('Clone failed: repo is private. "
          "EITHER make github.com/hugogobato/"
          "ccx-contextual-confounding public (Settings > General > "
          "Danger Zone), OR add a Colab Secret (key icon, left sidebar) "
          "named CCX_GH_PAT with a GitHub personal access token "
          "(repo scope), then rerun.')\n"
          "    r = os.system('git clone -q https://' + tok +\n"
          "                  '@github.com/hugogobato/'\n"
          "                  'ccx-contextual-confounding.git ' + REPO)\n"
          "    if r != 0:\n"
          "        raise RuntimeError('clone failed even with token; "
          "check PAT scope')\n"
          "    return True\n"
          "\n"
          "if not os.path.exists(REPO + '/.git'):\n"
          "    _clone()\n"
          "if not os.path.exists(REPO + '/src/enumeration.py'):\n"
          "    raise RuntimeError('checkout incomplete')\n"
          "os.system('git -C ' + REPO + ' checkout -q ' + GIT_SHA)\n"
          "if not os.path.exists(REPO + '/src/enumeration.py'):\n"
          "    raise RuntimeError('checkout incomplete after retry')\n"
          "sys.path.insert(0, REPO + '/src')\n"
          "import numpy as np\nimport pandas as pd\n")


def driver_from(name):
    return (ROOT / "scripts" / name).read_text()


def write_nb(tag, sid, cfg, setup_src, driver_src, path):
    cells = [cell_config(tag, sid, cfg),
             cell_setup(git_sha).replace(
                 "from continuous_witness import",
                 "pass\nfrom continuous_witness import")
             if False else cell_setup(git_sha),
             driver_src]
    Path(path).write_text(json.dumps(nb(cells)))


def main():
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    sha = current_sha()
    print("pinning GIT_SHA =", sha)
    seeds = json.loads((ROOT / "configs" / "seeds.json").read_text())
    p2 = seeds["phase2"]

    # ---------------- Phase 1 enumeration: 120 jobs -> 10 shards
    u = seeds["phase1_uniform"]
    bd = seeds["phase1_boundary"]
    jobs = [("u", (b, u["batch_size"], b))
            for b in range(u["n_batches"])]
    cid, remaining = 0, bd["n_points"]
    bchunk = max(u["batch_size"], 2000)
    while remaining > 0:
        take = min(bchunk, remaining)
        jobs.append(("b", (cid, take, bd["seed"] + cid)))
        remaining -= take
        cid += 1
    sp = seeds.get("phase1_sparse", {"n_points": 20000, "seed": 555000})
    cid, remaining = 0, sp["n_points"]
    while remaining > 0:
        take = min(2000, remaining)
        jobs.append(("s", (cid, take, sp["seed"] + cid)))
        remaining -= take
        cid += 1
    n_shard = 10
    per = (len(jobs) + n_shard - 1) // n_shard
    drv1 = driver_from("colab_driver_phase1.py")
    for s in range(n_shard):
        cfg = {"jobs": jobs[s * per:(s + 1) * per]}
        write_nb("phase1", s, cfg, cell_setup(sha), drv1,
                 SHARD_DIR / ("ccx_phase1_shard%02d.ipynb" % s))

    # ---------------- phase1 aggregate: 1 notebook
    write_nb("phase1agg", 0, {}, cell_setup(sha),
             driver_from("colab_driver_phase1_agg.py"),
             SHARD_DIR / "ccx_phase1agg.ipynb")

    # ---------------- WP 2.2: 180 groups -> 18 shards x 10
    groups22 = wp22_groups(p2, pilot=False)
    n_shard = 18
    per = (len(groups22) + n_shard - 1) // n_shard
    drv22 = driver_from("colab_driver_wp22.py")
    for s in range(n_shard):
        chunk = groups22[s * per:(s + 1) * per]
        for g in chunk:
            g.pop("cfg", None)
        cfg = {"groups": chunk, "cfg": p2}
        write_nb("wp22", s, cfg, cell_setup(sha), drv22,
                 SHARD_DIR / ("ccx_wp22_shard%02d.ipynb" % s))

    # ---------------- WP 2.3: 432 groups -> 9 shards x 48
    groups23 = wp23_groups(pilot=False)
    for g in groups23:
        g.pop("cfg", None)
    n_shard = 9
    per = (len(groups23) + n_shard - 1) // n_shard
    drv23 = driver_from("colab_driver_wp23.py")
    for s in range(n_shard):
        cfg = {"groups": groups23[s * per:(s + 1) * per]}
        write_nb("wp23", s, cfg, cell_setup(sha), drv23,
                 SHARD_DIR / ("ccx_wp23_shard%02d.ipynb" % s))

    # ---------------- WP 2.4: single
    write_nb("wp24", 0, {}, cell_setup(sha),
             driver_from("colab_driver_wp24.py"),
             SHARD_DIR / "ccx_wp24.ipynb")

    total = n_shard + 1 + 18 + 9 + 1
    print("wrote %d notebooks -> %s" % (total, SHARD_DIR))


if __name__ == "__main__":
    main()
