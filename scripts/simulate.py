"""Generate OpenFAST labels from hourly physical features."""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from windpower.simulation import Installation, SimulationCase, prepare_case, run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/simulation"))
    parser.add_argument("--installation", type=Path, default=Path("vendor"))
    for option in ("openfast", "turbsim", "controller", "templates", "turbsim-template"):
        parser.add_argument("--" + option, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    installation = Installation.from_prefix(args.installation)
    overrides = {name: getattr(args, name).resolve() for name in vars(installation) if getattr(args, name)}
    installation = replace(installation, **overrides)
    frame = pd.read_csv(args.features)
    if "time" not in frame:
        frame = frame.rename(columns={"target_time": "time"})
    frame["time"] = pd.to_datetime(frame.time, utc=True)
    frame = frame.sort_values("time")
    if frame.time.duplicated().any():
        raise ValueError("Simulation requires one ERA5 feature row per hour")
    if args.limit:
        frame = frame.head(args.limit)
    cases = [
        SimulationCase(
            case_id=str(row.get("case_id", f"era5_{row.time:%Y%m%dT%H%MZ}")), time=row.time.isoformat(),
            hub_wind_speed_mps=float(row.hub_wind_speed_mps), air_density_kgm3=float(row.air_density_kgm3),
            shear_exponent=float(row.shear_exponent), turbulence_intensity=float(row.get("turbulence_intensity", 0.08)),
            seed=int(row.get("seed", args.seed + i)),
        )
        for i, (_, row) in enumerate(frame.iterrows())
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        for case in cases:
            print(prepare_case(case, installation, args.output_dir))
        return
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for case, result in zip(cases, pool.map(lambda case: run_case(case, installation, args.output_dir), cases)):
            results.append({"time": case.time, "case_id": case.case_id, **{
                key: result[key] for key in ("target_power_kw", "power_std_kw", "half_mean_difference_kw")
            }})
            pd.DataFrame(results).to_csv(args.output_dir / "labels.csv", index=False)
            print(f"{case.case_id}: {result['target_power_kw']:.3f} kW", flush=True)


if __name__ == "__main__":
    main()
