"""Download the CAS Loss Reserving Database (NAIC Schedule P) into data/raw/.

Usage:
    python scripts/download_cas_data.py            # 1988-1997 data set (default)
    python scripts/download_cas_data.py --edition 98-07

Source and description:
https://www.casact.org/publications-research/research/research-resources/loss-reserving-data-pulled-naic-schedule-p
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://www.casact.org/sites/default/files"
EDITIONS = {
    "88-97": {
        line: f"{BASE}/2021-04/{line}_pos.csv"
        for line in ["ppauto", "comauto", "wkcomp", "medmal", "othliab", "prodliab"]
    },
    "98-07": {
        "ppauto": f"{BASE}/2026-03/ppauto_pos98-07%20%281%29.csv",
        **{
            line: f"{BASE}/2026-03/{line}_pos_98-07.csv"
            for line in ["comauto", "wkcomp", "medmal", "othliab", "prodliab"]
        },
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edition", choices=EDITIONS, default="88-97")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    out = args.out / args.edition
    out.mkdir(parents=True, exist_ok=True)
    for line, url in EDITIONS[args.edition].items():
        target = out / f"{line}.csv"
        if target.exists():
            print(f"skip {target} (exists)")
            continue
        print(f"download {line} ...")
        urllib.request.urlretrieve(url, target)
        print(f"  -> {target}")


if __name__ == "__main__":
    main()
