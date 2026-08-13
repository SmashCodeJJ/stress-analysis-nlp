#!/usr/bin/env python3
"""Download and merge Dreaddit CSV files from Hugging Face."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "dreaddit.csv"

URLS = [
    "https://huggingface.co/datasets/asmaab/dreaddit/resolve/main/train_data.csv",
    "https://huggingface.co/datasets/asmaab/dreaddit/resolve/main/validation_data.csv",
    "https://huggingface.co/datasets/asmaab/dreaddit/resolve/main/test-data.csv",
]


def main() -> int:
    import pandas as pd

    frames = []
    tmp_dir = ROOT / ".tmp_data"
    tmp_dir.mkdir(exist_ok=True)

    for i, url in enumerate(URLS):
        target = tmp_dir / f"part_{i}.csv"
        print(f"Downloading {url}")
        urlretrieve(url, target)
        frames.append(pd.read_csv(target))

    merged = pd.concat(frames, ignore_index=True)
    merged.to_csv(OUTPUT, index=False)
    print(f"Saved {len(merged)} rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
