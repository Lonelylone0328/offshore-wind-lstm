"""Training, prediction, evaluation and weather attribution commands."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import analysis, correction, correction_training, datasets, models, training
from .physics import NAMES, build_features


def feature_windows(path, kind):
    frame = pd.read_csv(path)
    for name in ("issue_time", "target_time"):
        frame[name] = pd.to_datetime(frame[name], utc=True)
    frame = frame.sort_values(["issue_time", "target_time"]).reset_index(drop=True)
    values = []
    for issue, group in frame.groupby("issue_time", sort=True):
        targets = pd.date_range(issue + pd.Timedelta(hours=1), periods=24, freq="h")
        if not pd.DatetimeIndex(group.target_time).equals(targets):
            raise ValueError("Each issue needs the complete consecutive t+1..24 sequence.")
        if kind == "gfs":
            available = pd.to_datetime(group.available_time, utc=True)
            cycles = pd.to_datetime(group.cycle_time, utc=True)
            if available.isna().any() or cycles.isna().any() or (available > issue).any():
                raise ValueError("GFS input must have been available at issue time.")
            if (cycles > issue - pd.Timedelta(hours=6)).any():
                raise ValueError("The GFS model requires a cycle at least six hours before issue.")
            if not (cycles + pd.to_timedelta(group.forecast_hour, unit="h")).eq(group.target_time).all():
                raise ValueError("Forecast cycle and lead do not match the target time.")
        values.append(group[NAMES].to_numpy(dtype=np.float32 if kind == "gfs" else float))
    return frame, np.stack(values)


def predict_command(args):
    folder = args.model_dir or args.models / args.model
    if args.model in ("era5", "gfs"):
        frame, x = feature_windows(args.input, args.model)
        power = models.predict(folder, x, args.model)
        columns = [c for c in ["issue_time", "target_time", "lead_hours"] if c in frame]
        result = frame[columns].copy()
        result["lead_hours"] = np.tile(np.arange(1, 25), len(power))
    else:
        data = correction.load_dataset(args.input) if args.input.suffix == ".npz" else correction.read_csv(args.input)
        power = correction.predict(folder, data, args.model, args.teacher or args.models / "era5")
        result = pd.DataFrame({"issue_time": np.repeat(data["issue_times"], 24),
                               "target_time": data["valid_times"][:, 24:].ravel(),
                               "lead_hours": np.tile(np.arange(1, 25), len(power))})
    result["power_kw"] = power.ravel()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    return {"output": str(args.output), "rows": len(result), "model": args.model}


def prepare_command(args):
    raw = pd.read_csv(args.input)
    if "time" not in raw:
        raw = raw.rename(columns={"valid_time": "time"})
    groups = raw.groupby("issue_time", sort=True) if "issue_time" in raw else [(None, raw)]
    outputs = []
    for issue, part in groups:
        if issue is not None:
            issue = pd.to_datetime(issue, utc=True)
        derived = build_features(part)
        if issue is not None:
            sea = part.loc[part.site_id.eq("sea")] if "site_id" in part else part
            sea = sea.copy()
            sea["time"] = pd.to_datetime(sea.time, utc=True)
            metadata = [c for c in ["cycle_time", "available_time", "forecast_hour"] if c in sea]
            derived = derived.merge(sea[["time", *metadata]], on="time", validate="one_to_one")
            derived = derived.rename(columns={"time": "target_time"})
            derived["issue_time"] = pd.Timestamp(issue)
            derived["lead_hours"] = ((derived.target_time - pd.Timestamp(issue)) / pd.Timedelta(hours=1)).astype(int)
            derived[NAMES] = derived[NAMES].astype(np.float32)
        outputs.append(derived)
    result = pd.concat(outputs, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    return {"output": str(args.output), "rows": len(result)}


def evaluate_corrections(args):
    data = correction.load_dataset(args.data)
    reference = pd.read_csv(args.labels)
    reference["time"] = pd.to_datetime(reference.time, utc=True).dt.tz_localize(None)
    truth = reference.set_index("time").loc[data["valid_times"][:, 24:].ravel()].target_power_kw.to_numpy()
    result = pd.DataFrame({"issue_time": np.repeat(data["issue_times"], 24),
                           "target_time": data["valid_times"][:, 24:].ravel(),
                           "lead_hours": np.tile(np.arange(1, 25), len(data["issue_times"])),
                           "target_power_kw": truth})
    metrics = {}
    for route in ("weather", "power"):
        prediction = correction.predict(args.models / route, data, route, args.models / "era5").ravel()
        result[route + "_power_kw"] = prediction
        metrics[route] = analysis.score(truth, prediction)
    args.output.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output / "predictions.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Offshore wind LSTM training and inference")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Calculate physical features from raw hourly sea weather")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("predict", help="Predict t+1..24 power")
    p.add_argument("--model", choices=["era5", "gfs", "weather", "power"], required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--teacher", type=Path)
    p = commands.add_parser("train", help="Train a selected model configuration")
    p.add_argument("--model", choices=["era5", "gfs", "weather", "power"], required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--targets", type=Path, default=Path("data/native_targets.npz"))
    p.add_argument("--teacher", type=Path, default=Path("models/era5"))
    p.add_argument("--epochs", type=int, help="Override maximum epochs")
    p = commands.add_parser("build-dataset", help="Align features and OpenFAST labels into training windows")
    p.add_argument("--model", choices=["era5", "gfs"], required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-segments", nargs="+", default=["winter", "spring"])
    p.add_argument("--validation-segments", nargs="+", default=["summer"])
    p = commands.add_parser("build-targets", help="Align ERA5 supervision to native GFS training windows")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--era5", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("evaluate", help="Recompute all six paired ERA5/GFS cells and decompositions")
    p.add_argument("--data", type=Path, default=Path("data/benchmark.npz"))
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    p = commands.add_parser("attribute", help="Analyze existing six-cell predictions")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("runs/attribution"))
    p = commands.add_parser("evaluate-corrections", help="Evaluate the two annual GFS correction models")
    p.add_argument("--data", type=Path, default=Path("data/native_test.npz"))
    p.add_argument("--labels", type=Path, default=Path("data/openfast_2024.csv"))
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--output", type=Path, default=Path("runs/corrections"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "predict":
        result = predict_command(args)
    elif args.command == "prepare":
        result = prepare_command(args)
    elif args.command == "build-dataset":
        result = datasets.build_dataset(args.model, args.features, args.labels, args.output,
                                        args.train_segments, args.validation_segments)
    elif args.command == "build-targets":
        correction.build_targets(args.data, args.era5, args.output)
        result = {"output": str(args.output)}
    elif args.command == "train":
        if args.model in ("era5", "gfs"):
            training.train(args.model, args.data, args.output, max_epochs=args.epochs)
        else:
            correction_training.train(args.model, args.data, args.targets, args.output,
                                      args.teacher, max_epochs=args.epochs)
        result = {"output": str(args.output), "model": args.model}
    elif args.command == "evaluate":
        result = analysis.evaluate(args.data, args.models, args.output)["overall"]
    elif args.command == "attribute":
        frame = pd.read_csv(args.input, float_precision="round_trip")
        result = analysis.attribute(frame, args.output)["overall"]
    else:
        result = evaluate_corrections(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
