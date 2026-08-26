import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from continuous_witness import k1_witness, hsic_resid_stat
import phase3_dgps as d3

d = pd.concat([pd.read_csv("results/raw/phase3/wp32_shard%02d.csv" % i)
               for i in range(12)], ignore_index=True)
print("total rows:", len(d))
print("cells:", d.groupby(["kind", "noise", "n", "d"]).ngroups,
      "| unique seeds per cell:")
cnt = d.groupby(["kind", "noise", "n", "d"])["seed"].nunique()
print(cnt.value_counts().to_dict())

# --- v2 semantics check: k1 obs must be per-trim (differs across trims)
ga = d[(d.method == "k1") & (d.noise == "gauss")]
piv = ga.pivot_table(index=["kind", "n", "d", "seed"], columns="trim",
                     values="stat_obs")
frac_same = float(np.mean(piv[0.00] == piv[0.01]))
print("k1 gauss rows with IDENTICAL obs at trim 0/0.01 (v1 leak):",
      frac_same)

# --- reproduce one full seed end-to-end from committed v2 code
r = d[(d.method == "k1") & (d.noise == "t3")].iloc[0]
rng = np.random.default_rng(int(r.seed))
x, y, _W = d3.sample_null(rng, int(r.n), int(r.d), noise=r.noise,
                          kind=r.kind)
ok1 = abs(k1_witness(x, y, trim_q=0.01) - r.stat_obs) < 1e-10
h = d[(d.method == "hsic") & (d.noise == "t3")].iloc[0]
rng2 = np.random.default_rng(int(h.seed))
x2, y2, _W = d3.sample_null(rng2, int(h.n), int(h.d), noise=h.noise,
                            kind=h.kind)
ok2 = abs(hsic_resid_stat(x2[:800], y2[:800]) - h.stat_obs) < 1e-10
print("k1 t3 trim0.01 reproduces:", ok1,
      "| hsic resid reproduces:", ok2)

# --- quick sizes at primary policy (the Defect-A payoff)
prim = {m: t for m, t in [("k1", .01), ("k2", .01), ("hsic", 0.)]}
sub = d[[abs(r.trim - prim.get(r.method, r.trim)) < 1e-9
         for r in d.itertuples()]]
rej = sub.assign(rej=lambda z: z["stat_obs"] > z["cv_0.05"])
sz = rej.groupby(["noise", "kind", "n", "d", "method"]).agg(
    size=("rej", "mean")).reset_index()
print("\nworst sizes by noise x method:")
print(sz.groupby(["noise", "method"])["size"].max().round(4).unstack())
