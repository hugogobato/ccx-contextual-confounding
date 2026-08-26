"""Generate self-contained Colab shard notebooks for WP 3.2 / WP 3.3.

Plan Section 8 requirements enforced per notebook: inline source (no repo
imports), SHARD_ID + CONFIG first cell, incremental appends after every
few seeds, safe download fallback, manifest row, resume logic.

Usage: python3 scripts/make_colab_shards.py
Outputs colab/shards/ccx_wp32_shardXX.ipynb (18x) and ccx_wp33_shardXX.ipynb
(16x). Upload to Colab accounts, Run all, download CSV+manifest into
results/raw/phase3/ (wp32/wp33 file names are already shard-unique).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_wp32_calibration import make_groups as wp32_groups  # noqa: E402
from run_wp33_separation import make_groups as wp33_groups  # noqa: E402

SHARD_DIR = ROOT / "colab" / "shards"


def read_src(name):
    return (ROOT / "src" / name).read_text()

CELL_DRIVER_WP32 = '''
# ---- WP 3.2 null-calibration driver ----
RESUME = os.path.exists(OUT)
done_seeds = set()
if RESUME:
    prev = pd.read_csv(OUT)
    done_seeds = set((r["kind"], r["noise"], r["n"], r["d"], r["seed"])
                     for _, r in prev.iterrows())
    print("resume:", len(done_seeds), "dataset-records found")

rows_all = []
for gi, g in enumerate(CONFIG["groups"]):
    n, d, noise, kind, B = g["n"], g["d"], g["noise"], g["kind"], g["B"]
    if noise == "gauss":
        bmap = dict((tq, (B if tq == 0.01 else min(B, 49)))
                    for tq in TRIMS)
        trims = TRIMS
    else:
        bmap = None
        trims = (0.01,)
    todo = [s for s in g["seeds"]
            if (kind, noise, n, d, s) not in done_seeds]
    print("[group %d/%d] n=%d d=%d %s %s: %d seeds"
          % (gi + 1, len(CONFIG["groups"]), n, d, noise, kind,
             len(todo)), flush=True)
    for j, seed in enumerate(todo):
        rng = np.random.default_rng(seed)
        x, y, _W = sample_null(rng, n, d, noise=noise, kind=kind)
        obs = {"k1": k1_witness(x, y),
               "k2": k2_witness(x, y),
               "hsic": hsic_stat(x[:HSIC_CAP], y[:HSIC_CAP])}
        boot = bootstrap_all(x, y, B, bmap, trims, seed)
        for meth in ("k1", "k2", "hsic"):
            for tq in (trims if meth != "hsic" else (0.0,)):
                if tq not in boot[meth]:
                    continue
                cvs = critical_values(boot[meth][tq], ALPHA_GRID)
                r = {"n": n, "d": d, "noise": noise, "kind": kind,
                     "seed": seed, "method": meth, "trim": tq,
                     "B": len(boot[meth][tq]), "stat_obs": obs[meth]}
                for a in ALPHA_GRID:
                    r["cv_%.2f" % a] = cvs[a]
                rows_all.append(r)
        if (j + 1) % 10 == 0 or (j + 1) == len(todo):
            pd.DataFrame(rows_all).to_csv(OUT, index=False)
            print("  %d/%d seeds, rows=%d" % (j + 1, len(todo),
                                              len(rows_all)), flush=True)
pd.DataFrame(rows_all).to_csv(OUT, index=False)

manifest = {"tag": TAG, "shard_id": SHARD_ID, "code_hash": CODE_HASH,
            "groups": len(CONFIG["groups"]),
            "rows_written": len(rows_all)}
mpath = os.path.join(os.path.dirname(OUT), "ccx_%s_manifest_shard%02d.json" % (TAG, SHARD_ID))
with open(mpath, "w") as fh:
    json.dump(manifest, fh, indent=2)
print("MANIFEST:", json.dumps(manifest))

try:
    from google.colab import files
    files.download(OUT)
    files.download(mpath)
    print("Downloaded:", OUT)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
'''

CELL_DRIVER_WP33 = '''
# ---- WP 3.3 separation-study driver (statistics only; CVs applied locally)
RESUME = os.path.exists(OUT)
done_keys = set()
if RESUME:
    prev = pd.read_csv(OUT)
    done_keys = set((r["kind"], r["b"], r["n"], r["seed"])
                    for _, r in prev.iterrows())
    print("resume:", len(done_keys), "records found")

rows_all = []
for gi, g in enumerate(CONFIG["groups"]):
    n, d, b, kind, noise = g["n"], g["d"], g["b"], g["kind"], g["noise"]
    todo = [s for s in g["seeds"]
            if (kind, b, n, s) not in done_keys]
    print("[group %d/%d] n=%d d=%d b=%g %s %s: %d seeds"
          % (gi + 1, len(CONFIG["groups"]), n, d, b, noise, kind,
             len(todo)), flush=True)
    for j, seed in enumerate(todo):
        rng = np.random.default_rng(seed)
        x, y, _W = sample_confounded(rng, n, d, b, noise=noise, kind=kind)
        obs = {"k1": k1_witness(x, y, trim_q=0.01),
               "k2": k2_witness(x, y, trim_q=0.01),
               "hsic": hsic_stat(x[:HSIC_CAP], y[:HSIC_CAP])}
        for meth, sv in obs.items():
            rows_all.append({"n": n, "d": d, "noise": noise, "kind": kind,
                             "b": b, "seed": seed, "method": meth,
                             "stat_obs": sv})
        if (j + 1) % 25 == 0 or (j + 1) == len(todo):
            pd.DataFrame(rows_all).to_csv(OUT, index=False)
            print("  %d/%d" % (j + 1, len(todo)), flush=True)
pd.DataFrame(rows_all).to_csv(OUT, index=False)

manifest = {"tag": TAG, "shard_id": SHARD_ID, "code_hash": CODE_HASH,
            "groups": len(CONFIG["groups"]),
            "rows_written": len(rows_all)}
mpath = os.path.join(os.path.dirname(OUT), "ccx_%s_manifest_shard%02d.json" % (TAG, SHARD_ID))
with open(mpath, "w") as fh:
    json.dump(manifest, fh, indent=2)
print("MANIFEST:", json.dumps(manifest))

try:
    from google.colab import files
    files.download(OUT)
    files.download(mpath)
    print("Downloaded:", OUT)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
'''

def nb(cells):
    out_cells = []
    for c in cells:
        out_cells.append({"cell_type": "code", "execution_count": None,
                          "metadata": {}, "outputs": [],
                          "source": c.splitlines(keepends=True)})
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"accelerator": "None",
                         "colab": {"provenance": []}},
            "cells": out_cells}


def build(tag, config, setup_src, driver_src, path):
    sid = config["_shard_id"]
    cfg_json = json.dumps(config)
    cell1 = ("# CCX %s shard %02d - config cell\n"
             "import os\n"
             "BASE = '/content' if os.path.isdir('/content') else '.'\n"
             "TAG = '%s'\nSHARD_ID = %d\n"
             "OUT = BASE + '/ccx_%s_shard%02d.csv'\n"
             "CONFIG_JSON = %r\n"
             "import json as _json\nCONFIG = _json.loads(CONFIG_JSON)\n"
             % (tag.upper(), sid, tag, sid, tag, sid, cfg_json))
    cell2 = ("import os, sys, time, json, hashlib\n"
             "import numpy as np\nimport pandas as pd\n"
             "CODE_HASH = hashlib.sha256((%r).encode())"
             ".hexdigest()[:16]\nprint(TAG, 'shard', SHARD_ID)"
             % (setup_src + driver_src,) + "\n" + setup_src)
    nb_ = nb([cell1, cell2, driver_src])
    Path(path).write_text(json.dumps(nb_))


def main():
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    cfg32 = json.loads(
        (ROOT / "configs" / "seeds.json").read_text())["phase2"]

    cw = read_src("continuous_witness.py")
    cw = cw.replace("from witnesses import kl_em_batch\n", "")
    cw = cw.replace("from models import build_iv_A_general, "
                    "_mixed_radix_tables\n", "")
    p3 = read_src("phase3_dgps.py").replace(
        "from models import build_iv_A_general, _mixed_radix_tables\n", "")
    p3 = p3.replace("from continuous_witness import k1_witness, "
                    "k1_multiplier_bootstrap\n", "")

    setup_common = ("import os, sys, time, json, hashlib\n"
                    "import numpy as np\nimport pandas as pd\n\n"
                    + cw + "\n" + p3 +
                    "\n\ndef critical_values(stat_draws, alpha_grid):\n"
                    "    s = np.sort(np.asarray(stat_draws, dtype=float))\n"
                    "    out = {}\n"
                    "    for a in alpha_grid:\n"
                    "        idx = min(max(int(np.ceil((1.0 - a) * len(s)))"
                    " - 1, 0), len(s) - 1)\n"
                    "        out[float(a)] = float(s[idx])\n"
                    "    return out\n\n\n"
                    "TRIMS = (0.0, 0.01, 0.05)\n"
                    "ALPHA_GRID = [round(0.01 * a, 2) for a in "
                    "range(1, 21)]\n"
                    "HSIC_CAP = 800\n")

    def bootstrap_fn():
        return (
            "\ndef bootstrap_all(x, y, B, bmap, trims, seed):\n"
            "    _, k1d = k1_multiplier_bootstrap(\n"
            "        x, y, B=B, trim_grid=trims,\n"
            "        rng=np.random.default_rng(seed + 7000000), bmap=bmap)\n"
            "    _, k2d = k2_multiplier_bootstrap(\n"
            "        x, y, B=B, trim_grid=trims,\n"
            "        rng=np.random.default_rng(seed + 7100000), bmap=bmap)\n"
            "    xc, yc = x[:HSIC_CAP], y[:HSIC_CAP]\n"
            "    hb = hsic_resid_permutation(xc, yc, B=B,\n"
            "                                rng=np.random.default_rng("
            "seed + 7200000))\n"
            "    return {'k1': k1d, 'k2': k2d, 'hsic': {0.0: hb}}\n")

    # ---------------- WP 3.2: 54 groups -> 18 shards x 3 groups
    groups32 = wp32_groups(cfg32, pilot=False)
    n_shard = 18
    per = (len(groups32) + n_shard - 1) // n_shard
    drv32 = (Path(ROOT) / "scripts" / "colab_driver_wp32.py").read_text()
    drv33 = (Path(ROOT) / "scripts" / "colab_driver_wp33.py").read_text()
    for s in range(n_shard):
        chunk = groups32[s * per:(s + 1) * per]
        for g in chunk:
            g.pop("cfg", None)
        config = {"_shard_id": s, "groups": chunk}
        setup = (setup_common.replace("__SOURCE_STAMP__",
                                      setup_common + drv32)
                 + bootstrap_fn())
        build("wp32", config, setup, drv32,
              SHARD_DIR / f"ccx_wp32_shard{s:02d}.ipynb")

    # ---------------- WP 3.3: 810 groups -> 16 shards
    n33 = 16
    groups33 = wp33_groups(pilot=False)
    per = (len(groups33) + n33 - 1) // n33
    for s in range(n33):
        chunk = groups33[s * per:(s + 1) * per]
        config = {"_shard_id": s, "groups": chunk}
        setup = (setup_common.replace("__SOURCE_STAMP__",
                                      setup_common + drv33)
                 )
        build("wp33", config, setup, drv33,
              SHARD_DIR / f"ccx_wp33_shard{s:02d}.ipynb")

    print(f"wrote {n_shard} wp32 + {n33} wp33 notebooks -> {SHARD_DIR}")


if __name__ == "__main__":
    main()
