"""Download ERA5 or issued GFS point weather."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from windpower.acquisition import SITES, download_era5, download_gfs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="source", required=True)
    for source in ("era5", "gfs"):
        sub = subparsers.add_parser(source)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--sites", type=Path, help="CSV: site_id,latitude,longitude")
        if source == "era5":
            sub.add_argument("--start", required=True, help="UTC ISO date/time")
            sub.add_argument("--end", required=True)
            sub.add_argument("--snapshot")
        else:
            sub.add_argument("--issue-times", type=Path, required=True, help="CSV containing issue_time")
            sub.add_argument("--resolution", type=float, choices=(0.5, 0.25), default=0.5)
            sub.add_argument("--history-hours", type=int, choices=(0, 24), default=0)
            sub.add_argument("--neighbors", action="store_true", help="18 native 0.25-degree points")
            sub.add_argument("--cache", type=Path, default=Path("data/download-cache/gfs"))
            sub.add_argument("--npz-output", type=Path)
            sub.add_argument("--workers", type=int, default=4)
            sub.add_argument("--buffer-minutes", type=int)
            sub.add_argument("--minimum-cycle-age", type=int)
    args = parser.parse_args()
    sites = SITES
    if args.sites:
        sites = list(pd.read_csv(args.sites)[["site_id", "latitude", "longitude"]].itertuples(index=False, name=None))
    if args.source == "era5":
        download_era5(pd.to_datetime(args.start, utc=True), pd.to_datetime(args.end, utc=True),
                      args.output, sites, args.snapshot)
    else:
        if args.neighbors:
            if args.resolution != 0.25 or args.sites:
                parser.error("--neighbors requires --resolution 0.25 and the default sites")
            sites = [
                (name if (dy, dx) == (0, 0) else f"{name}_{dy:+g}_{dx:+g}", lat + dy, lon + dx)
                for name, lat, lon in SITES for dy in (-0.25, 0, 0.25) for dx in (-0.25, 0, 0.25)
            ]
        issues = pd.read_csv(args.issue_times).issue_time
        download_gfs(
            issues, args.output, args.cache, args.resolution, sites, args.history_hours, args.workers,
            args.buffer_minutes, args.minimum_cycle_age, args.npz_output,
        )
    print(args.output)


if __name__ == "__main__":
    main()
