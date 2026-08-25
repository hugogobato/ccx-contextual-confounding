"""Generate THIN Colab shard notebooks for WP 3.2 / WP 3.3.

cell1 config | cell2 clone pinned SHA + imports | cell3 driver.
Drivers live in scripts/colab_driver_wp32.py / colab_driver_wp33.py
(single source of truth; this generator reads them at build time).
Reproducibility via pinned GIT_SHA recorded in each shard manifest.

Usage: python3 scripts/make_colab_shards.py
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from run_wp32_calibration import make_groups as wp32_groups
from run_wp33_separation import make_groups as wp33_groups

SHARD_DIR = ROOT / "colab" / "shards"
GIT_URL = "https://github.com/hugogobato/ccx-contextual-confounding.git"
GIT_SHA = "a643aff94fe71ba26b14b04c7f7126514e1f2b7b"


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


def build(tag, sid, groups, driver_src):
    cfg_json = json.dumps({"groups": groups})
    cell1 = (
        "# CCX %s shard %02d - config cell\n"
        "TAG = '%s'\nSHARD_ID = %d\n"
        "OUT = '/content/ccx_%s_shard%02d.csv'\n"
        "CONFIG_JSON = %r\n"
        "import json as _json\nCONFIG = _json.loads(CONFIG_JSON)\n"
        % (tag.upper(), sid, tag, sid, tag, sid, cfg_json))
    cell2 = (
        "import os, sys\n"
        "REPO = '/content/ccx-src'\n"
        "GIT_URL = '%s'\nGIT_SHA = '%s'\n" % (GIT_URL, GIT_SHA)
        + "if not os.path.isdir(REPO):\n"
          "    r = os.system('git clone -q ' + GIT_URL + ' ' + REPO)\n"
          "    if r != 0:\n"
          "        try:\n"
          "            from google.colab import userdata\n"
          "            tok = userdata.get('CCX_GH_PAT')\n"
          "            r = os.system('git clone -q https://' + tok +\n"
          "                          '@github.com/hugogobato/'\n"
          "                          'ccx-contextual-confounding.git '\n"
          "                          + REPO)\n"
          "        except Exception:\n"
          "            r = 1\n"
          "    if r != 0:\n"
          "        raise RuntimeError('clone failed: repo is private - "
          "add a Colab Secret named CCX_GH_PAT (GitHub PAT, repo scope) "
          "and rerun')\n"
          "os.system('git -C ' + REPO + ' checkout -q ' + GIT_SHA)\n"
          "sys.path.insert(0, REPO + '/src')\n"
          "import numpy as np\nimport pandas as pd\n"
          "from continuous_witness import (k1_witness,"
          " k1_multiplier_bootstrap,\n"
          "                                k2_witness,"
          " k2_multiplier_bootstrap,\n"
          "                                hsic_stat, hsic_bootstrap)\n"
          "from phase3_dgps import sample_null, sample_confounded\n"
          "from calibration import critical_values\n"
          "TRIMS = (0.0, 0.01, 0.05)\n"
          "ALPHA_GRID = [round(0.01 * a, 2) for a in range(1, 21)]\n"
          "HSIC_CAP = 2500\n")
    return nb([cell1, cell2, driver_src])


def main():
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    drv32 = (ROOT / "scripts" / "colab_driver_wp32.py").read_text()
    drv33 = (ROOT / "scripts" / "colab_driver_wp33.py").read_text()

    cfg32 = json.loads(
        (ROOT / "configs" / "seeds.json").read_text())["phase2"]
    groups32 = wp32_groups(cfg32, pilot=False)
    for g in groups32:
        g.pop("cfg", None)
    n32, n33 = 18, 16
    per = (len(groups32) + n32 - 1) // n32
    for s in range(n32):
        nb_ = build("wp32", s, groups32[s * per:(s + 1) * per], drv32)
        (SHARD_DIR / ("ccx_wp32_shard%02d.ipynb" % s)).write_text(
            json.dumps(nb_))

    groups33 = wp33_groups(pilot=False)
    per = (len(groups33) + n33 - 1) // n33
    for s in range(n33):
        nb_ = build("wp33", s, groups33[s * per:(s + 1) * per], drv33)
        (SHARD_DIR / ("ccx_wp33_shard%02d.ipynb" % s)).write_text(
            json.dumps(nb_))

    print("wrote %d wp32 + %d wp33 notebooks -> %s" %
          (n32, n33, SHARD_DIR))


if __name__ == "__main__":
    main()
