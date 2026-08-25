"""WP 2.2 null calibration suite driver.

Design (predeclared; deviations D2/D3 logged in memos/gate_B_memo.md):
- Null kinds: chain, u_on_x, u_on_y (fresh random IV-form SCM per
  replication, 200 seeds) + anchor_deterministic / anchor_interior (fixed
  laws, 100 seeds). inert_u dropped: distributionally identical to chain.
- Arms: clean + 5% epsilon-contaminated. Bootstrap calibration runs on the
  CLEAN arm only; contaminated arms are evaluated (stat_obs) against pooled
  clean critical values. Contaminated POPULATION laws are classified
  feasible / infeasible; feasible-contamination rows enter worst-case size,
  infeasible ones are detection evidence.
- Engines para_boot / crt_cond / subsample; adaptive decisions use
  own-draw CVs. Subsampling CVs carry the sqrt(m/n) rate correction.
- Cell-class triage:
    binary  (2-2-2): engines P/C/S, B=199, B_lp=199, all n, boot subset
                     {60 random / 30 anchor}
    mid     (2-2-3),(2-3-3),(3-3-3),(2-2-5): engines P/C/S for closed-form
                     stats, split/cf and LP stats under P only, B=99,
                     B_lp per config, all n, boot subset {24/12}
    big     (2-2-8),(3-3-5): engines P/C for kl_plugin/batteries/gtest,
                     split+LP under P with reduced draws, EM iters 120,
                     n in {500,2000,8000}, boot subset {24/12}

Row schema: one row per (dataset-arm, stat, engine): stat_obs + cv_<a> cols;
B=0 rows are observed-only (excluded from CV pooling).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from models import build_iv_A_general  # noqa: E402
from witnesses import kl_em_batch  # noqa: E402
from witness_estimators import (counts_to_conditionals,  # noqa: E402
                                kl_split_crossfit_batch, cf1_plugin_stat,
                                slack_plugin_stat)
from calibration import draw_bootstrap_counts, critical_values, fit_null_q  # noqa: E402
import phase2_dgps as dg  # noqa: E402
from baselines.ineq_tests import (inflation_order2_feasible,  # noqa: E402
                                  build_gm_battery, gtest_independence_stat)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "phase2_discrete"
RAW = ROOT / "results" / "raw" / "phase2"

ALPHA_GRID = [round(0.01 * a, 2) for a in range(1, 21)]
ENGINE_OFFSET = {"para_boot": 1, "crt_cond": 2, "subsample": 3}


def cell_class(cell_id):
    if cell_id == "2-2-2":
        return "binary"
    if cell_id in ("2-2-8", "3-3-5"):
        return "big"
    return "mid"


class Triage:
    def __init__(self, cfg, cell_id):
        cls = cell_class(cell_id)
        self.cls = cls
        c = cfg["B_cells"]
        lp = cfg["B_lp_cells"]
        sub = cfg["boot_subset_binary"] if cls == "binary" \
            else cfg["boot_subset_nonbinary"]
        self.B = c.get(cell_id, c["default"])
        self.B_lp = lp.get(cell_id, lp["default"])
        self.boot_random = sub["random"]
        self.boot_anchor = sub["anchor"]
        if cls == "big":
            self.em_iters = 120
            self.fit_iter = 80
            self.split_engines = ("para_boot",)
            self.lp_engines = ("para_boot",)
            self.kl_engines = ("para_boot", "crt_cond")
            self.closed_engines = ("para_boot", "crt_cond")
            self.ns = [500, 2000, 8000]
        elif cls == "mid":
            self.em_iters = 250
            self.fit_iter = 150
            self.split_engines = ("para_boot",)
            self.lp_engines = ("para_boot",)
            self.kl_engines = ("para_boot", "crt_cond", "subsample")
            self.closed_engines = ("para_boot", "crt_cond", "subsample")
            self.ns = None          # all
        else:
            self.em_iters = 250
            self.fit_iter = 150
            self.split_engines = ("para_boot",)
            self.lp_engines = ("para_boot", "crt_cond", "subsample")
            self.kl_engines = ("para_boot", "crt_cond", "subsample")
            self.closed_engines = ("para_boot", "crt_cond", "subsample")
            self.ns = None


def load_cfg():
    return json.loads((ROOT / "configs" / "seeds.json").read_text())["phase2"]


def _battery_path(cell):
    return RES / f"gm_battery_{'-'.join(map(str, cell))}.npz"


def get_gm_battery(M, cell):
    p = _battery_path(cell)
    if p.exists():
        z = np.load(p)
        return z["W"], z["c"]
    bat = build_gm_battery(M, rng_seed=424242)
    RES.mkdir(parents=True, exist_ok=True)
    W = np.array([w for w, _ in bat])
    c = np.array([cc for _, cc in bat])
    np.savez(p, W=W, c=c)
    return W, c


def get_binary_facets():
    z = np.load(ROOT / "results" / "phase1_enumeration" / "facets_iv.npz")
    return z["H"].astype(float), z["b"].astype(float)


def observed_stats(M, Mint, counts, kz, kx, ky, facW, facC, Wg, cg):
    cond = counts_to_conditionals(counts[None, :], kz)[0]
    out = {}
    val, _ = kl_em_batch(M.astype(float), cond[None, :])
    out["kl_plugin"] = float(val[0])
    sp, cf = kl_split_crossfit_batch(M, counts[None, :].astype(np.int64),
                                     kz, np.random.default_rng(0xC0FFEE),
                                     chunk=1)
    out["kl_split"], out["kl_crossfit"] = float(sp[0]), float(cf[0])
    out["cf1_margin"] = float(max(0.0, 1.0 - cf1_plugin_stat(Mint, counts,
                                                             kz)))
    out["slack_plugin"] = float(slack_plugin_stat(Mint, counts, kz))
    out["gm_battery"] = float(np.max(np.maximum(cond @ Wg.T - cg, 0.0)))
    if facW is not None:
        out["pearl_facet"] = float(np.max(np.maximum(cond @ facW.T - facC,
                                                     0.0)))
    out["gtest"] = gtest_independence_stat(counts, kz, kx, ky)
    return out


def bootstrap_rows(M, Mint, counts, kz, kx, ky, seed, cfg, tr, facW, facC,
                   Wg, cg):
    blk = kx * ky
    q_hat, push = fit_null_q(M, counts_to_conditionals(counts[None, :],
                                                       kz)[0])
    n_tot = int(counts.sum())
    m_sub = max(2, min(int(np.floor(n_tot ** cfg["subsampling_exponent"])),
                       n_tot))
    rate_sub = np.sqrt(m_sub / float(n_tot))

    draws_by_key = {}
    for eng in ENGINE_OFFSET:
        brng = np.random.default_rng(
            (seed * 1000 + cfg["bootstrap_seed_offset"] +
             ENGINE_OFFSET[eng]) % (2 ** 31))
        boot_counts = draw_bootstrap_counts(push.reshape(kz, blk), counts,
                                            kz, brng, tr.B, eng,
                                            cfg["subsampling_exponent"])
        boot_cond = counts_to_conditionals(boot_counts, kz)
        scale = rate_sub if eng == "subsample" else 1.0

        if eng in tr.kl_engines:
            val_p, _ = kl_em_batch(M.astype(float), boot_cond,
                                   n_iter=tr.em_iters, tol=1e-9)
            draws_by_key[f"kl_plugin|{eng}"] = np.asarray(val_p) * scale
        if eng in tr.split_engines:
            B_sc = min(tr.B, 99) if M.shape[1] > 500 else tr.B
            sp, cfr = kl_split_crossfit_batch(
                M, boot_counts[:B_sc], kz,
                np.random.default_rng((seed * 100000 + 17) % (2 ** 31)),
                fit_iter=tr.fit_iter, fit_tol=1e-9)
            draws_by_key[f"kl_split|{eng}"] = sp * scale
            draws_by_key[f"kl_crossfit|{eng}"] = cfr * scale
        if eng in tr.lp_engines:
            cf_v, sl_v = [], []
            for b in range(min(tr.B_lp, len(boot_counts))):
                cf_v.append(max(0.0, 1.0 -
                                cf1_plugin_stat(Mint, boot_counts[b], kz)))
                sl_v.append(slack_plugin_stat(Mint, boot_counts[b], kz))
            draws_by_key[f"cf1_margin|{eng}"] = np.array(cf_v) * scale
            draws_by_key[f"slack_plugin|{eng}"] = np.array(sl_v) * scale

        if eng in tr.closed_engines:
            vg = np.max(np.maximum(boot_cond @ Wg.T - cg[None, :], 0.0),
                        axis=1)
            draws_by_key[f"gm_battery|{eng}"] = vg * scale
            if facW is not None:
                vf = np.max(np.maximum(boot_cond @ facW.T - facC[None, :],
                                       0.0), axis=1)
                draws_by_key[f"pearl_facet|{eng}"] = vf * scale
            gt = np.array([gtest_independence_stat(boot_counts[b],
                                                   kz, kx, ky)
                           for b in range(len(boot_counts))])
            draws_by_key[f"gtest|{eng}"] = gt * scale

    obs = observed_stats(M, Mint, counts, kz, kx, ky, facW, facC, Wg, cg)
    rows = []
    for key, draws in draws_by_key.items():
        statname, eng = key.split("|")
        cvs = critical_values(draws, ALPHA_GRID)
        row = {"stat": statname, "engine": eng, "B": int(len(draws)),
               "stat_obs": obs.get(statname, np.nan)}
        for a in ALPHA_GRID:
            row[f"cv_{a:.2f}"] = cvs[a]
        rows.append(row)
    return rows


def process_group(group):
    kz, kx, ky = group["cell"]
    cell_id = "-".join(map(str, group["cell"]))
    n, kind = group["n"], group["kind"]
    cfg = group["cfg"]
    M = build_iv_A_general(kz, kx, ky)
    Mint = M.astype(int)
    blk = kx * ky
    eps = cfg["contamination_eps"]
    tr = Triage(cfg, cell_id)

    facW = facC = None
    if (kz, kx, ky) == (2, 2, 2):
        facW, facC = get_binary_facets()
    Wg, cg = get_gm_battery(M, (kz, kx, ky))

    popc_fixed = None
    if kind == "anchor_interior":
        popc_fixed = dg.anchor_interior_conditional(M, kz, kx, ky, seed=1234)
    elif kind == "anchor_deterministic":
        popc_fixed = np.zeros((kz, blk))
        for z in range(kz):
            x, y = z % kx, (z % kx) % ky
            popc_fixed[z, x * ky + y] = 1.0

    out_rows = []
    t0 = time.perf_counter()
    boot_set = set(group["boot_seeds"])
    for seed in group["seeds"]:
        rng = np.random.default_rng(seed)
        if popc_fixed is not None:
            rows = dg.sample_rows_from_conditional(rng, n, popc_fixed, kz,
                                                   kx, ky) \
                if kind == "anchor_interior" else \
                dg.anchor_deterministic_rows(kz, kx, ky, n)
            popc = popc_fixed
        else:
            rows, meta = dg.sample_rows_null(rng, n, kz, kx, ky, kind)
            popc = meta["pop_cond"]

        arms = [("clean", rows, False)]
        rc = np.random.default_rng(seed + cfg["contamination_seed_offset"])
        arms.append(("contam", dg.contaminate_rows(rc, rows, kx, ky, eps),
                     True))

        do_boot_clean = seed in boot_set
        for arm, rws, is_contam in arms:
            counts = dg.counts_from_rows(rws, kz, kx, ky)
            feas_contam = ""
            if is_contam:
                pc = dg.contaminated_population_conditional(popc, eps)
                f2, _dt = inflation_order2_feasible(M, pc.reshape(-1))
                feas_contam = bool(f2)

            base = {"cell": cell_id, "kind": kind, "n": n, "seed": seed,
                    "arm": arm, "feas_contam": feas_contam}

            if is_contam or not do_boot_clean:
                obs = observed_stats(M, Mint, counts, kz, kx, ky,
                                     facW, facC, Wg, cg)
                for statname, sv in obs.items():
                    out_rows.append(dict(
                        base, stat=statname, engine="none", B=0,
                        stat_obs=sv,
                        **{f"cv_{a:.2f}": np.nan for a in ALPHA_GRID}))
                continue

            brows = bootstrap_rows(M, Mint, counts, kz, kx, ky, seed, cfg,
                                   tr, facW, facC, Wg, cg)
            for r in brows:
                out_rows.append(dict(base, **r))

    group["t_group_s"] = time.perf_counter() - t0
    return out_rows


def make_groups(cfg, cells, kinds, pilot=False):
    groups = []
    ns = cfg["null_seeds"]
    for cell in cells:
        cell_id = "-".join(map(str, cell))
        tr = Triage(cfg, cell_id)
        ns_list = ([250, 2000] if pilot else
                   (tr.ns if tr.ns else cfg["n_grid"]))
        for kind in kinds:
            seed_list = list(range(ns["start"],
                                   ns["start"] + ns["count"]))
            anchor = kind.startswith("anchor")
            if pilot:
                seed_list = seed_list[:6]
            elif anchor:
                seed_list = seed_list[:100]
            nb = tr.boot_anchor if anchor else tr.boot_random
            if pilot:
                boot_seeds = seed_list
            else:
                boot_seeds = seed_list[:nb]
            for nv in ns_list:
                groups.append({"cell": tuple(cell), "kind": kind, "n": nv,
                               "seeds": seed_list, "boot_seeds": boot_seeds,
                               "cfg": cfg})
    return groups


def run_one_group(g):
    fn = RAW / (f"wp22_{'-'.join(map(str, g['cell']))}_{g['kind']}_"
                f"n{g['n']}.csv")
    if fn.exists():                      # resume support
        return g["cell"], g["kind"], g["n"], -1, 0.0
    rows = process_group(g)
    df = pd.DataFrame(rows)
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = str(fn) + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, fn)
    return g["cell"], g["kind"], g["n"], len(rows), g.get("t_group_s", 0)


def safe_run_one_group(g):
    try:
        return run_one_group(g)
    except Exception as ex:
        return ("ERR", f"{type(ex).__name__}: {str(ex)[:140]}",
                "-".join(map(str, g["cell"])), g["kind"], g["n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--cells", default="")
    ap.add_argument("--kinds", default="")
    ap.add_argument("--dump-groups", action="store_true",
                    help="write group index JSON and exit")
    ap.add_argument("--one", type=int, default=-1,
                    help="process a single group index and exit")
    args = ap.parse_args()
    cfg = load_cfg()
    cells = ([tuple(map(int, c.split("-"))) for c in args.cells.split(",")]
             if args.cells else [tuple(c) for c in cfg["alphabet_cells"]])
    kinds = (args.kinds.split(",") if args.kinds
             else cfg["wp22_null_kinds"])

    groups = make_groups(cfg, cells, kinds, pilot=args.pilot)

    if args.dump_groups:
        out = ROOT / "configs" / "phase2_groups.json"
        serial = [{**g, "cell": list(g["cell"])} for g in groups]
        out.write_text(json.dumps(serial))
        print(f"wrote {len(groups)} groups -> {out}")
        return

    if args.one >= 0:
        res = safe_run_one_group(groups[args.one])
        print(res, flush=True)
        return

    print(f"groups: {len(groups)} "
          f"(datasets {sum(len(g['seeds']) for g in groups)}, "
          f"boot-clean-datasets "
          f"{sum(len(g['boot_seeds']) for g in groups)})", flush=True)
    t_start = time.time()

    import multiprocessing as mp
    nw = int(os.environ.get("CCX_WORKERS", "6"))
    done = 0
    err_keys = []
    with mp.Pool(processes=nw, maxtasksperchild=2) as pool:
        for res in pool.imap_unordered(safe_run_one_group, groups,
                                       chunksize=1):
            done += 1
            el = time.time() - t_start
            if res[0] == "ERR":
                err_keys.append((tuple(int(x) for x in res[2].split("-")),
                                 res[3], res[4]))
                print(f"[wp22] JOB ERROR {done}/{len(groups)}: {res}",
                      flush=True)
            else:
                cell, kind, nv, nr, tg = res
                print(f"[wp22] {done}/{len(groups)} "
                      f"{'-'.join(map(str, cell))} {kind} n={nv}: "
                      f"{nr} rows, {tg:.0f}s | elapsed {el:.0f}s",
                      flush=True)

    failed = [g for g in groups
              if ((g["cell"], g["kind"], g["n"]) in err_keys)]
    if failed:
        print(f"[wp22] retrying {len(failed)} failed groups serially",
              flush=True)
        for g in failed:
            try:
                fn = RAW / (f"wp22_{'-'.join(map(str, g['cell']))}_"
                            f"{g['kind']}_n{g['n']}.csv")
                if fn.exists():
                    fn.unlink()
                out = run_one_group(g)
                print(f"[wp22-retry] ok: {g['cell']} {g['kind']} n={g['n']}"
                      f" ({out[3]} rows)", flush=True)
            except Exception as ex:
                print(f"[wp22-retry] STILL FAILING {g['cell']} "
                      f"{g['kind']} n={g['n']}: {ex}", flush=True)

    print("driver complete")


if __name__ == "__main__":
    main()
