# ---- WP 3.3 separation-study driver (statistics only; CVs applied locally)
from continuous_witness import (k1_witness, k1_multiplier_bootstrap,
                                k2_witness, k2_multiplier_bootstrap, hsic_stat, hsic_bootstrap)
from phase3_dgps import sample_null, sample_confounded
from calibration import critical_values

RESUME = os.path.exists(OUT)
done_keys = set()
if RESUME:
    try:
        prev = pd.read_csv(OUT)
        done_keys = set((r["kind"], r["b"], r["n"], r["seed"])
                        for _, r in prev.iterrows())
        print("resume:", len(done_keys), "records found")
    except Exception:
        print("resume: unreadable partial file, starting fresh")
        RESUME = False

rows_all = []
for gi, g in enumerate(CONFIG["groups"]):
    n, d, b, kind, noise = g["n"], g["d"], g["b"], g["kind"], g["noise"]
    todo = [s for s in g["seeds"] if (kind, b, n, s) not in done_keys]
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

manifest = {"tag": TAG, "shard_id": SHARD_ID, "git_sha": GIT_SHA,
            "groups": len(CONFIG["groups"]),
            "rows_written": len(rows_all)}
mpath = os.path.join(os.path.dirname(OUT),
                     "ccx_%s_manifest_shard%02d.json"
                     % (TAG, SHARD_ID))
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
