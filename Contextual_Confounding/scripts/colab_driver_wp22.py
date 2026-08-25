# WP 2.2 null-calibration driver (groups from CONFIG; cfg embedded)
import copy

CFG = CONFIG["cfg"]
for g in CONFIG["groups"]:
    g["cfg"] = CFG
    g["boot_seeds"] = g["boot_seeds"]

from run_wp22_calibration import process_group, run_one_group
from pathlib import Path as _P
import pandas as pd

rows_all = []
for gi, g in enumerate(CONFIG["groups"]):
    rows = process_group(g)
    df = pd.DataFrame(rows)
    fn = "/content/ccx_wp22_%s_%s_n%d.csv" % (
        "-".join(map(str, g["cell"])), g["kind"], g["n"])
    df.to_csv(fn, index=False)
    rows_all.append({"group": gi, "file": fn.split("/")[-1],
                     "rows": len(rows)})
    print("[group %d/%d] %s %s n=%d: %d rows"
          % (gi + 1, len(CONFIG["groups"]),
             "-".join(map(str, g["cell"])), g["kind"], g["n"],
             len(rows)), flush=True)

manifest_path = "/content/ccx_wp22_manifest_shard%02d.json" % SHARD_ID
manifest = {"tag": TAG, "shard_id": SHARD_ID, "git_sha": GIT_SHA,
            "files": rows_all}
with open(manifest_path, "w") as fh:
    json.dump(manifest, fh, indent=2)
print("MANIFEST:", json.dumps(manifest))

try:
    from google.colab import files
    for r in rows_all:
        files.download("/content/" + r["file"])
    files.download(manifest_path)
except Exception as e:
    print("(Not on Colab / download skipped):", e)
