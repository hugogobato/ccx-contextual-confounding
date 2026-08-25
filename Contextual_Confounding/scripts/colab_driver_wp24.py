# WP 2.4 scaling pain map (single-notebook job)
from run_wp24_scaling import main
main()

import shutil, json
src = "/home/content"  # placeholder no-op
out = "results/phase2_discrete/lp_walltime_map.csv"
shutil.copyfile(out, "/content/ccx_wp24_lp_walltime_map.csv")

manifest = {"tag": TAG, "shard_id": SHARD_ID, "git_sha": GIT_SHA,
            "rows_written": "see csv"}
with open("/content/ccx_wp24_manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)
print("MANIFEST:", json.dumps(manifest))

try:
    from google.colab import files
    files.download("/content/ccx_wp24_lp_walltime_map.csv")
    files.download("/content/ccx_wp24_manifest.json")
except Exception as e:
    print("(Not on Colab / download skipped):", e)
