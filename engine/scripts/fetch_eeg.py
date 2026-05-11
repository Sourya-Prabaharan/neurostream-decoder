"""Download a small subset of PhysioNet eegmmidb for the streaming pipeline.

Default: subject S001 runs 3 and 4 (~5 MB total). The runs use the
BCI2000 motor-imagery paradigm (left/right fist movement vs rest).

Usage:
    python -m engine.scripts.fetch_eeg
    python -m engine.scripts.fetch_eeg --subjects 1 2 --runs 3 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.neurostream.realdata import download_run, PHYSIONET_BASE


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects", type=int, nargs="+", default=[1])
    p.add_argument("--runs", type=int, nargs="+", default=[3, 4])
    p.add_argument("--cache", default=str(ROOT / "data" / "raw_eeg"))
    args = p.parse_args()

    cache = Path(args.cache)
    print(f"PhysioNet source: {PHYSIONET_BASE}")
    print(f"Cache dir       : {cache}")
    for sub in args.subjects:
        for run in args.runs:
            dst = download_run(sub, run, cache)
            print(f"  ok  S{sub:03d}R{run:02d}.edf  ({dst.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
