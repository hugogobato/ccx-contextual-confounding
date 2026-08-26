"""Aggregate WP2.3 shard CSVs (ccx_wp23_shard*.csv) → power_curves.csv."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "phase2"

def load():
    shards = sorted(RAW.glob("ccx_wp23_shard*.csv"))
    legacy = sorted(RAW.glob("wp23_*.csv"))
    files = shards + legacy
    if not files:
        raise FileNotFoundError(f"no WP2.3 files in {RAW}")
    print(f"found {len(shards)} shards + {len(legacy)} legacy")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def main():
    df = load()
    print(f"total rows {len(df)}")
    tmp = RAW / "wp23_combined_for_agg.csv"
    df.to_csv(tmp, index=False)
    print(f"wrote {tmp}")
    try:
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        import aggregate_wp23
        if hasattr(aggregate_wp23, "main"):
            aggregate_wp23.main()
        tmp.unlink(missing_ok=True)
    except Exception as e:
        print(f"aggregate_wp23 failed: {e}")
        import traceback; traceback.print_exc()
        print(f"combined file kept at {tmp} — handle manually")

if __name__ == "__main__":
    main()
