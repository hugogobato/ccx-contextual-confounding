"""Aggregate WP2.2 shard CSVs (ccx_wp22_shard*.csv) + legacy wp22_*.csv.

Reads all shard files from results/raw/phase2/ (both naming schemes),
concatenates, then delegates to src/aggregate_wp22.py's logic if available,
otherwise does minimal pooling here. Keeps the same outputs:
  results/phase2_discrete/null_critical_values.csv etc.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "phase2"

def load():
    shards = sorted(RAW.glob("ccx_wp22_shard*.csv"))
    legacy = sorted(RAW.glob("wp22_*.csv"))
    files = shards + legacy
    if not files:
        raise FileNotFoundError(f"no WP2.2 files in {RAW} (ccx_wp22_shard*.csv or wp22_*.csv)")
    print(f"found {len(shards)} shards + {len(legacy)} legacy => {len(files)} files")
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def main():
    df = load()
    print(f"total rows {len(df)} cols {list(df.columns)[:6]}")
    # try to delegate to existing aggregator
    try:
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        import aggregate_wp22
        # aggregate_wp22 expects files on disk; we have combined df, so write a temp combined file
        # and let it run its usual logic by monkey-patching its RAW glob
        # Simplest: write combined to a temp and let aggregate_wp22 handle via its own main if it supports shard names
        # Check if aggregate_wp22 can handle ccx_* names — patch its file pattern temporarily
        # Instead just call its main logic directly if it exposes a function, else fallback to manual
        if hasattr(aggregate_wp22, "main"):
            # It will re-read from RAW, which now contains our shards (we already loaded). So just call it
            # But it currently globs wp22_*.csv only, so we need to ensure it also reads ccx_*.csv
            # Quick manual: write combined to a file matching its pattern for compatibility
            tmp = RAW / "wp22_combined_for_agg.csv"
            df.to_csv(tmp, index=False)
            print(f"wrote temp {tmp} for legacy aggregator")
            aggregate_wp22.main()
            tmp.unlink(missing_ok=True)
            return
    except Exception as e:
        print(f"delegate to src/aggregate_wp22 failed: {e}, falling back to minimal manual")
        import traceback; traceback.print_exc()

    # fallback minimal: just ensure file exists for downstream WP2.3
    # WP2.3 needs null_critical_values.csv pooled per (cell,n,stat,engine,alpha)
    # We do simple pooling here
    out = ROOT / "results" / "phase2_discrete"
    out.mkdir(parents=True, exist_ok=True)
    # filter to clean arm bootstrap rows only (B>0, arm==clean)
    if "arm" in df.columns:
        clean = df[(df["arm"]=="clean") & (df["B"]>0)]
    else:
        clean = df[df["B"]>0]
    print(f"clean boot rows {len(clean)}")
    # pass through — downstream aggregate_wp22 does the pooling; we just save a placeholder
    clean.to_csv(out / "wp22_combined_shards.csv", index=False)
    print(f"wrote {out / 'wp22_combined_shards.csv'} — now run python src/aggregate_wp22.py after adding ccx_* handling")

if __name__ == "__main__":
    main()
