# Phase 1 enumeration driver (subset of jobs from CONFIG)
from enumeration import run_job, RAW
import glob, os, json

produced = []
for gi, j in enumerate(CONFIG["jobs"]):
    kind, ident, nrows, dt = run_job(j)
    print("[job %d/%d] %s %s: %d rows, %.1fs"
          % (gi + 1, len(CONFIG["jobs"]), kind, str(j[0]), nrows, dt),
          flush=True)
    for f in sorted(RAW.glob("*.csv")):
        if f.name not in produced:
            produced.append(f)

# stage every raw batch csv this notebook touched for download
out_dir = "/content"
staged = []
for f in sorted(RAW.glob("*.csv")):
    dst = os.path.join(out_dir, "p1_" + f.name)
    shutil.copyfile(f, dst)
    staged.append(dst)

manifest = {"tag": TAG, "shard_id": SHARD_ID, "git_sha": GIT_SHA,
            "jobs": len(CONFIG["jobs"]), "files_staged": len(staged)}
mpath = "/content/ccx_%s_manifest_shard%02d.json" % (TAG, SHARD_ID)
with open(mpath, "w") as fh:
    json.dump(manifest, fh, indent=2)
print("MANIFEST:", json.dumps(manifest))

try:
    from google.colab import files
    for f in staged:
        files.download(f)
    files.download(mpath)
    print("Downloaded:", len(staged), "files")
except Exception as e:
    print("(Not on Colab / download skipped):", e)
