# Phase 1 aggregation notebook (run AFTER all phase1 shards downloaded+merged)
# Expects: repo clone present; results/raw/phase1/*.csv populated with ALL
# uniform batches 000..099, boundary chunks, sparse chunks.
import glob
import pandas as pd
import numpy as np

from models import build_iv_A, iv_vertices, ROOT
from enumeration import _load_facets, process_batch

files = sorted(glob.glob(str(ROOT / "results/raw/phase1/*.csv")))
print("raw batch files:", len(files))
assert len(files) >= 119, "incomplete raw set - download all phase1 shards first"
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

V = iv_vertices().astype(float)
df_v = pd.DataFrame(process_batch(V, "vertex", -1, -1))
df = pd.concat([df, df_v], ignore_index=True)

seeds = json.loads(open(ROOT / "configs/seeds.json").read_text())
u = seeds["phase1_uniform"]
sub = seeds.get("phase1_inflation_subsample",
                {"n_points": 20000})
ksub = max(1, (u["n_batches"] * u["batch_size"]) // sub["n_points"])
mask = ((df["source"] == "uniform") & (df["batch"] % ksub == 0)) | \
       ((df["source"] == "boundary") & (df["batch"] % 5 == 0))
idx_sub = df.index[mask].to_numpy()[:sub["n_points"]]
print("inflation-2 subsample:", len(idx_sub))
A16 = build_iv_A()
from witnesses import iv_inflation2_feasible
inf_res = []
for i in idx_sub:
    e = df.loc[i, ["e_%d" % j for j in range(8)]].to_numpy(dtype=float)
    inf_res.append(iv_inflation2_feasible(e, A16))
df.loc[idx_sub, "inflation2_feasible"] = np.array(inf_res, dtype=float)

RES = ROOT / "results/phase1_enumeration"
RES.mkdir(parents=True, exist_ok=True)
df.to_csv(RES / "t1_dictionary.csv", index=False)

ctx = df[(~df["lp_feasible"]) & (~df["borderline_band"])].copy()
pd.DataFrame({
    "instance_id": ctx["instance_id"],
    "strongly_contextual": ctx["cf1_hard"] < 1.0,
    "cf1_hard": ctx["cf1_hard"],
    "maximally_contextual": ctx["maximally_contextual"].astype(bool),
    "cf1_soft": ctx["cf1_soft"],
    "degree_honest": ctx["slack_l1"] / 2.0,
    "degree_signed": ctx["degree_signed"],
    "kl_contextuality": ctx["kl_contextuality"],
    "slack_l1": ctx["slack_l1"],
}).to_csv(RES / "hierarchy_placement.csv", index=False)

strict = pd.DataFrame({
    "instance_id": ctx["instance_id"],
    "nm_equalities_hold": True,
    "n_nm_constraints": 0,
    "margin_cf1_soft": 1.0 - ctx["cf1_soft"],
    "margin_degree_honest": ctx["slack_l1"] / 2.0,
    "margin_kl": ctx["kl_contextuality"],
})
strict["strict_instance"] = (strict["margin_cf1_soft"] >= 0.05) | \
                            (strict["margin_degree_honest"] >= 0.05) | \
                            (strict["margin_kl"] >= 0.05)
strict.to_csv(RES / "strictness_scan.csv", index=False)

from scipy.stats import spearmanr
wit_cols = ["cf1_soft", "degree_signed", "kl_contextuality", "cf1_hard"]
rows_r = []
d_all = df[df["slack_l1"].notna()]
d_ctx2 = d_all[~d_all["lp_feasible"] & ~d_all["borderline_band"]]
def safe_spearman(a, col):
    if len(a) < 3 or a[col].nunique() < 2 or a["slack_l1"].nunique() < 2:
        return np.nan
    return float(spearmanr(a[col], a["slack_l1"]).statistic)
for wcol in wit_cols:
    dd = d_all[[wcol, "slack_l1"]].replace([np.inf], np.nan).dropna()
    rows_r.append({"witness": wcol,
                   "spearman_rho": safe_spearman(dd, wcol),
                   "spearman_rho_contextual":
                       safe_spearman(d_ctx2.dropna(subset=[wcol]), wcol),
                   "n": int(len(dd)), "monotone_injective_flag": False})
pd.DataFrame(rows_r).to_csv(RES / "witness_lp_redundancy.csv", index=False)

agree = int(df["agree_c1a"].sum()); nb = int((~df["borderline_band"]).sum())
print("==== PHASE 1 REGEN SUMMARY ====")
print("instances:", len(df), "| C1a agreement: %d/%d = %.4f%%"
      % (agree, nb, 100.0 * agree / max(nb, 1)))
print("contextual:", int(len(ctx)))
manifest = {"tag": "phase1agg", "git_sha": GIT_SHA,
            "rows_dictionary": len(df)}
with open("/content/ccx_phase1agg_manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)

try:
    from google.colab import files
    for f in ("t1_dictionary.csv", "hierarchy_placement.csv",
              "strictness_scan.csv", "witness_lp_redundancy.csv"):
        files.download(str(RES / f))
except Exception as e:
    print("(Not on Colab / download skipped):", e)
