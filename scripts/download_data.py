"""Download the public IBM telco churn sample into data/raw.

The Kaggle page is the canonical citation. This script uses the CSV published
in the IBM telco-customer-churn-on-icp4d repository because that file is a
direct download. If the download fails, place Telco-Customer-Churn.csv in
data/raw yourself. Do not substitute an unlabeled synthetic file.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def main() -> int:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    if DESTINATION.exists() and DESTINATION.stat().st_size > 100_000:
        print(f"Already present: {DESTINATION}")
        return 0
    print(f"Downloading {URL}")
    try:
        urllib.request.urlretrieve(URL, DESTINATION)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "Place the IBM Telco Customer Churn CSV at data/raw/Telco-Customer-Churn.csv.\n"
            "Source page: https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
            file=sys.stderr,
        )
        return 1
    print(f"Saved {DESTINATION} ({DESTINATION.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
