"""Phase 4 Block A: discrete grid at scale (null calibration, power,
adversarial nulls) on the k_Y in {4,5,8} x (kz,kx)=(2,2) ladder.

Arms (configs/phase4.json, committed before runs):
- null : fresh IV-form SCMs (chain/u_on_x/u_on_y) x n {500,1000,2000} x
         tail {none, t3}; tail=t3 is the predeclared BOUNDARY-STRESS arm
         (p_z ~ Dirichlet(0.3), D-P4.1). 200 seeds, bootstrap subset 24,
         engines {para_boot, crt_cond} (D-P4.4). Raw rows -> p4dnull_*.csv.
- alt  : mixture family (primary strength family of WP 2.3) at rho
         {0.2,0.4,0.6,0.8}; tail=t3 uses near-vertex anchors
         (MixtureCell anchor_conc=0.3, D-P4.1). Point statistics only.
         Raw rows -> p4dalt_*.csv.
- adv  : 4 adversarial null configs (A1-A4) judged against envelope CVs
         from the null arm (aggregator). Raw rows -> p4dadv_*.csv.

Envelope CV rule (Gate B D6): per (cell, n, tail, stat, engine), max over
null kinds of the mean own-draw CV; computed by src/aggregate_phase4.py.
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
from run_wp22_calibration import (load_cfg as load_cfg_phase2,  # noqa: E402
                                  Triage, observed_stats, bootstrap_rows,
                                  get_gm_battery)
from baselines.ineq_tests import inflation_order2_feasible  # noqa: E402
import phase2_dgps as dg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "phase4"
ALPHA_GRID = [round(0.01 * a, 2) for a in range(1, 21)]
ENGINES = ("para_boot", "crt_cond")
P3_CONC = 0.3          # boundary-stress Dirichlet concentration (D-P4.1)
ALT_SEED_TAIL_NONE = 20260825   # MixtureCell default (WP 2.3 family)
ALT_SEED_TAIL_T3 = 20260826     # predeclared boundary-anchor stream


def load_p4_cfg():
    p4 = json.loads((ROOT / "configs" / "phase4.json").read_text())
    seeds = json.loads((ROOT / "configs" / "seeds.json").read_text())
    return p4, seeds["phase4"], load_cfg_phase2()


def p4_triage(cfg2, cfg4, cell_id):
    """WP 2.2 Triage with Phase 4 B_lp overrides and the reduced engine set."""
    tr = Triage(cfg2, cell_id)
    tr.B = cfg4["bootstrap_B"]
    tr.B_lp = cfg4["B_lp_cells"].get(cell_id, tr.B_lp)
    tr.kl_engines = ENGINES
    tr.closed_engines = ENGINES
    tr.split_engines = ("para_boot",)
    tr.lp_engines = ("para_boot",)
    return tr


# --------------------------------------------------------------- null arm

def process_null_group(g):
    kz, kx, ky = g["cell"]
    cell_id = "-".join(map(str, g["cell"]))
    M = build_iv_A_general(kz, kx, ky)
    Mint = M.astype(int)
    tr = g["tr"]
    Wg, cg = get_gm_battery(M, (kz, kx, ky))
    p_conc = P3_CONC if g["tail"] == "t3" else None
    out = []
    for seed in g["seeds"]:
        rng = np.random.default_rng(seed)
        rows, meta = dg.sample_rows_null(rng, n=g["n"], kz=kz, kx=kx, ky=ky,
                                         kind=g["kind"], p_conc=p_conc)
        counts = dg.counts_from_rows(rows, kz, kx, ky)
        base = {"cell": cell_id, "kind": g["kind"], "tail": g["tail"],
                "n": g["n"], "seed": seed, "arm": "clean",
                "feas_contam": ""}
        if seed in g["boot_seeds"]:
            brows = bootstrap_rows(M, Mint, counts, kz, kx, ky, seed,
                                   g["cfg2"], tr, None, None, Wg, cg)
            for r in brows:
                if r["engine"] in ENGINES:
                    out.append(dict(base, **r))
        else:
            obs = observed_stats(M, Mint, counts, kz, kx, ky, None, None,
                                 Wg, cg)
            for statname, sv in obs.items():
                out.append(dict(base, stat=statname, engine="none", B=0,
                                stat_obs=sv,
                                **{f"cv_{a:.2f}": np.nan
                                   for a in ALPHA_GRID}))
    return out


def make_null_groups(p4, cfg2, cfg4, pilot=False):
    dg_ = p4["discrete_grid"]
    seeds = list(range(700000, 700000 + dg_["reps"]))
    if pilot:
        seeds, boot = seeds[:6], seeds[:3]
    else:
        boot = seeds[:dg_["boot_subset_random"]]
    groups = []
    for cell in dg_["cells"]:
        tr = p4_triage(cfg2, cfg4, "-".join(map(str, cell)))
        ns = dg_["n_grid"][:2] if pilot else dg_["n_grid"]
        for n in ns:
            for kind in dg_["null_kinds"]:
                for tail in dg_["tails"]:
                    groups.append({"cell": tuple(cell), "n": n,
                                   "kind": kind, "tail": tail,
                                   "seeds": seeds, "boot_seeds": boot,
                                   "tr": tr, "cfg2": cfg2})
    return groups


# ---------------------------------------------------------------- alt arm

def process_alt_group(g):
    kz, kx, ky = g["cell"]
    cell_id = "-".join(map(str, g["cell"]))
    M = build_iv_A_general(kz, kx, ky)
    Mint = M.astype(int)
    Wg, cg = get_gm_battery(M, (kz, kx, ky))
    mix = dg.MixtureCell(kz, kx, ky,
                         alt_seed=(ALT_SEED_TAIL_NONE
                                   if g["tail"] == "none"
                                   else ALT_SEED_TAIL_T3),
                         anchor_conc=(1.0 if g["tail"] == "none"
                                      else P3_CONC))
    out = []
    for seed in g["seeds"]:
        rng = np.random.default_rng(seed)
        rows = mix.sample_rows(rng, g["n"], g["rho"])
        counts = dg.counts_from_rows(rows, kz, kx, ky)
        t0 = time.perf_counter()
        obs = observed_stats(M, Mint, counts, kz, kx, ky, None, None,
                             Wg, cg)
        dt = time.perf_counter() - t0
        for statname, sv in obs.items():
            out.append({"cell": cell_id, "tail": g["tail"],
                        "family": "mixture", "rho": g["rho"], "n": g["n"],
                        "seed": seed, "dt_stats_s": dt,
                        "stat": statname, "engine": "none",
                        "stat_obs": sv})
    return out


def make_alt_groups(p4, pilot=False):
    dg_ = p4["discrete_grid"]
    seeds = list(range(800000, 800000 + dg_["reps"]))
    if pilot:
        seeds = seeds[:6]
    groups = []
    for cell in dg_["cells"]:
        ns = dg_["n_grid"][:2] if pilot else dg_["n_grid"]
        rhos = dg_["rho_grid"][:2] if pilot else dg_["rho_grid"]
        for n in ns:
            for rho in rhos:
                for tail in dg_["tails"]:
                    groups.append({"cell": tuple(cell), "n": n, "rho": rho,
                                   "tail": tail, "seeds": seeds})
    return groups


# --------------------------------------------------------- adversarial arm

def _det_nonlinear_rows(rng, n, kz, kx, ky):
    z = rng.integers(0, kz, size=n)
    x = (z + 1) % kx
    y = (3 * x ** 2 + 2 * z + 1) % ky
    return np.stack([z, x, y], axis=1)


def process_adv_group(g):
    kz, kx, ky = g["cell"]
    cell_id = "-".join(map(str, g["cell"]))
    M = build_iv_A_general(kz, kx, ky)
    Mint = M.astype(int)
    Wg, cg = get_gm_battery(M, (kz, kx, ky))
    blk = kx * ky
    out = []
    for seed in g["seeds"]:
        rng = np.random.default_rng(seed)
        feas_contam = ""
        if g["adv_id"] == "A1_two_null_mixture":
            # Shared context mass p_z: the observable 50/50 mixture then has
            # EQUAL per-z mixture weights, so its per-context conditionals
            # are the pushforward of the coupling mixture (feasible).
            while True:
                p_z = rng.dirichlet(np.ones(kz))
                if np.all((p_z > 0.10) & (p_z < 0.90)):
                    break
            s1, _ = dg.sample_null_mechanism(rng, kz, kx, ky, "chain",
                                             p_z_given=p_z)
            s2, _ = dg.sample_null_mechanism(rng, kz, kx, ky, "chain",
                                             p_z_given=p_z)

            def _draw(samp, m):
                z = rng.choice(kz, size=m, p=p_z)
                u = rng.integers(0, 2, size=m)
                xy = np.array([samp(int(zz), int(uu))
                               for zz, uu in zip(z, u)])
                return np.stack([z, xy[:, 0], xy[:, 1]], axis=1)

            rows = np.concatenate(
                [_draw(s1, g["n"] // 2), _draw(s2, g["n"] - g["n"] // 2)],
                axis=0)
        elif g["adv_id"] == "A2_det_nonlinear_chain":
            rows = _det_nonlinear_rows(rng, g["n"], kz, kx, ky)
        elif g["adv_id"] == "A3_bursty_u_on_y":
            rows, _ = dg.sample_rows_null(rng, g["n"], kz, kx, ky,
                                          "u_on_y", p_conc=P3_CONC)
        elif g["adv_id"] == "A4_feasible_contamination":
            rows, meta = dg.sample_rows_null(rng, g["n"], kz, kx, ky,
                                             "chain")
            rc = np.random.default_rng(seed + 104729)
            rows = dg.contaminate_rows(rc, rows, kx, ky, eps=0.02)
            pc = dg.contaminated_population_conditional(meta["pop_cond"],
                                                        0.02)
            f2, _dt = inflation_order2_feasible(M, pc.reshape(-1))
            feas_contam = bool(f2)
        else:
            raise ValueError(g["adv_id"])
        counts = dg.counts_from_rows(rows, kz, kx, ky)
        obs = observed_stats(M, Mint, counts, kz, kx, ky, None, None,
                             Wg, cg)
        for statname, sv in obs.items():
            out.append({"cell": cell_id, "adv_id": g["adv_id"],
                        "tail_env": g["tail_env"], "n": g["n"],
                        "seed": seed, "feas_contam": feas_contam,
                        "stat": statname, "engine": "none",
                        "stat_obs": sv})
    return out


def make_adv_groups(p4, pilot=False):
    seeds = list(range(900000, 900000 + p4["adversarial_nulls"]["reps"]))
    if pilot:
        seeds = seeds[:6]
    groups = []
    for spec in p4["adversarial_nulls"]["discrete"]:
        groups.append({"adv_id": spec["id"], "cell": tuple(spec["cell"]),
                       "n": spec["n"], "tail_env": spec["tail_env"],
                       "seeds": seeds})
    return groups


# ------------------------------------------------------------------ driver

def run_one(fn, g, proc):
    if fn.exists():
        return -1
    rows = proc(g)
    df = pd.DataFrame(rows)
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = str(fn) + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, fn)
    return len(rows)


_JOB = {}


def _run_idx(i):
    g, proc, fn_of = _JOB["groups"][i], _JOB["proc"], _JOB["fn_of"]
    return run_one(fn_of(g), g, proc)


def run_all(tag, groups, proc, fn_of):
    _JOB.update(groups=groups, proc=proc, fn_of=fn_of)
    print(f"[{tag}] groups: {len(groups)}", flush=True)
    import multiprocessing as mp
    nw = int(os.environ.get("CCX_WORKERS", "6"))
    t0 = time.time()
    done = 0
    with mp.Pool(processes=nw, maxtasksperchild=4) as pool:
        for nr in pool.imap_unordered(_run_idx, range(len(groups)),
                                      chunksize=1):
            done += 1
            print(f"[{tag}] {done}/{len(groups)} rows={nr} "
                  f"| {time.time()-t0:.0f}s", flush=True)
    print("driver complete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["null", "alt", "adv"])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--one", type=int, default=-1)
    ap.add_argument("--dump-groups", action="store_true")
    args = ap.parse_args()
    p4, cfg4, cfg2 = load_p4_cfg()
    if args.arm == "null":
        groups = make_null_groups(p4, cfg2, cfg4, pilot=args.pilot)
        proc, tag = process_null_group, "p4dnull"
        fn_of = lambda g: (RAW / (f"p4dnull_{'-'.join(map(str, g['cell']))}"
                                  f"_{g['kind']}_{g['tail']}_n{g['n']}.csv"))
    elif args.arm == "alt":
        groups = make_alt_groups(p4, pilot=args.pilot)
        proc, tag = process_alt_group, "p4dalt"
        fn_of = lambda g: (RAW / (f"p4dalt_{'-'.join(map(str, g['cell']))}"
                                  f"_{g['tail']}_rho{g['rho']:.1f}"
                                  f"_n{g['n']}.csv"))
    else:
        groups = make_adv_groups(p4, pilot=args.pilot)
        proc, tag = process_adv_group, "p4dadv"
        fn_of = lambda g: RAW / f"p4dadv_{g['adv_id']}.csv"

    if args.dump_groups:
        serial = [{**{k: v for k, v in g.items()
                      if k in ("cell", "n", "kind", "tail", "rho",
                               "adv_id", "tail_env")},
                   "cell": list(g["cell"]),
                   "n_seeds": len(g["seeds"])} for g in groups]
        (ROOT / "configs" / "phase4_discrete_groups.json").write_text(
            json.dumps(serial))
        print(f"wrote {len(groups)} {tag} groups")
        return
    if args.one >= 0:
        print(run_one(fn_of(groups[args.one]), groups[args.one], proc),
              flush=True)
        return

    run_all(tag, groups, proc, fn_of)


if __name__ == "__main__":
    main()
