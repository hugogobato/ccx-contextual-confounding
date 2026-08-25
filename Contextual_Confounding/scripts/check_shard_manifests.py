"""Mechanical completeness check for downloaded Colab shards.
Drop downloaded CSVs + manifest JSONs into colab/downloads/ then run:
    python3 scripts/check_shard_manifests.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "colab" / "downloads"


def main():
    ok, bad = [], []
    man = sorted(DL.glob("*manifest*.json"))
    if not man:
        print("no manifests found in", DL)
        raise SystemExit(1)
    import pandas as pd
    for m in man:
        info = json.loads(m.read_text())
        tag, sid = info["tag"], info["shard_id"]
        csv = DL / ("ccx_%s_shard%02d.csv" % (tag, sid))
        if not csv.exists():
            bad.append((tag, sid, "csv missing"))
            continue
        n = len(pd.read_csv(csv))
        if n == info["rows_written"]:
            ok.append((tag, sid, n))
        else:
            bad.append((tag, sid,
                        "rows %d != manifest %d" % (n,
                                                    info["rows_written"])))
    print("complete: %d | problems: %d" % (len(ok), len(bad)))
    for b in bad:
        print("PROBLEM:", b)
    raise SystemExit(0 if not bad else 2)


if __name__ == "__main__":
    main()
