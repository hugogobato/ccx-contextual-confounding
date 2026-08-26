# ---- WP 3.2 null-calibration driver (v2: per-trim obs + residual-HSIC) ----
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
        obs = {("k1", tq): k1_witness(x, y, trim_q=tq) for tq in trims}
        obs.update({("k2", tq): k2_witness(x, y, trim_q=tq)
                    for tq in trims})
        xc, yc = x[:HSIC_CAP], y[:HSIC_CAP]
        obs_hsic = hsic_resid_stat(xc, yc)
        boot = bootstrap_all(x, y, B, bmap, trims, seed)
        for meth in ("k1", "k2"):
            for tq in trims:
                if tq not in boot[meth]:
                    continue
                cvs = critical_values(boot[meth][tq], ALPHA_GRID)
                r = {"n": n, "d": d, "noise": noise, "kind": kind,
                     "seed": seed, "method": meth, "trim": tq,
                     "B": len(boot[meth][tq]),
                     "stat_obs": obs[(meth, tq)]}
                for a in ALPHA_GRID:
                    r["cv_%.2f" % a] = cvs[a]
                rows_all.append(r)
        cvs = critical_values(boot["hsic"][0.0], ALPHA_GRID)
        r = {"n": n, "d": d, "noise": noise, "kind": kind,
             "seed": seed, "method": "hsic", "trim": 0.0,
             "B": len(boot["hsic"][0.0]), "stat_obs": obs_hsic}
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
mpath = "/content/ccx_%s_manifest_shard%02d.json" % (TAG, SHARD_ID)
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
