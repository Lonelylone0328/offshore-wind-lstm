"""Six-cell paired evaluation and algebraic error decomposition."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .models import predict
from .physics import NAMES

KEYS = ["segment_id", "issue_time", "target_time", "lead_hours"]
CELLS = {
    "B_E": ("empirical", "ERA5"), "B_G": ("empirical", "GFS"),
    "F_EE": ("ERA5_trained_LSTM", "ERA5"), "F_EG": ("ERA5_trained_LSTM", "GFS"),
    "F_GE": ("GFS_trained_LSTM", "ERA5"), "F_GG": ("GFS_trained_LSTM", "GFS"),
}


def score(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("Reference and prediction arrays must have the same nonempty shape.")
    error = predicted - actual
    mse = float(np.mean(error**2))
    return {"points": int(error.size), "mse_kw2": mse, "rmse_kw": float(np.sqrt(mse)),
            "mae_kw": float(np.mean(np.abs(error))), "bias_kw": float(np.mean(error))}


def improvement(model_rmse, baseline_rmse):
    return None if baseline_rmse <= 1e-12 else 100 * (1 - model_rmse / baseline_rmse)


def mse_decomposition(actual, era5, gfs):
    actual, era5, gfs = (np.asarray(a, dtype=float) for a in (actual, era5, gfs))
    residual, shift = era5 - actual, gfs - era5
    a, b = float(np.mean(residual**2)), float(np.mean((residual + shift)**2))
    d, interaction = float(np.mean(shift**2)), float(2 * np.mean(residual * shift))
    return {"era5_mse_kw2": a, "gfs_mse_kw2": b, "mse_increase_kw2": b - a,
            "source_shift_mse_kw2": d, "source_shift_rmse_kw": float(np.sqrt(d)),
            "interaction_kw2": interaction, "closure_residual_kw2": b - a - d - interaction}


def gain_decomposition(rates):
    if any(rates[k] is None for k in ("F_EE", "F_EG", "F_GE", "F_GG")):
        return {"defined": False}
    ee, eg, ge, gg = (rates[k] for k in ("F_EE", "F_EG", "F_GE", "F_GG"))
    weather, model = ((ee - eg) + (ge - gg)) / 2, ((ee - ge) + (eg - gg)) / 2
    gap = ee - gg
    return {"defined": True, "total_gap_pp": gap, "weather_source_pp": weather,
            "frozen_model_pp": model, "weather_effect_with_era5_model_pp": ee - eg,
            "weather_effect_with_gfs_model_pp": ge - gg, "model_effect_with_era5_pp": ee - ge,
            "model_effect_with_gfs_pp": eg - gg, "factor_interaction_pp": ee - eg - ge + gg,
            "weather_share_of_gap_percent": None if abs(gap) < 1e-6 else 100 * weather / gap,
            "model_share_of_gap_percent": None if abs(gap) < 1e-6 else 100 * model / gap,
            "closure_residual_pp": gap - weather - model}


def denominator_decomposition(be, le, bg, lg):
    if min(be, bg) <= 1e-12:
        return {"defined": False}
    ae, ag = be - le, bg - lg
    savings = 50 * (ae - ag) * (1 / be + 1 / bg)
    denominator = 50 * (1 / be - 1 / bg) * (ae + ag)
    gap = 100 * (ae / be - ag / bg)
    return {"defined": True, "era5_absolute_saving_kw": ae, "gfs_absolute_saving_kw": ag,
            "absolute_saving_change_pp": savings, "baseline_denominator_change_pp": denominator,
            "total_gap_pp": gap, "closure_residual_pp": gap - savings - denominator}


def analyze_frame(frame):
    truth = frame.target_power_kw.to_numpy()
    metrics = {cell: {"model": model, "weather": weather, **score(truth, frame[cell])}
               for cell, (model, weather) in CELLS.items()}
    for metric in metrics.values():
        baseline = metrics["B_E" if metric["weather"] == "ERA5" else "B_G"]["rmse_kw"]
        metric["rmse_saving_kw"] = baseline - metric["rmse_kw"]
        metric["same_weather_improvement_percent"] = improvement(metric["rmse_kw"], baseline)
    rates = {cell: m["same_weather_improvement_percent"] for cell, m in metrics.items()}
    mse = {model: mse_decomposition(truth, frame[left], frame[right])
           for model, left, right in [("empirical", "B_E", "B_G"),
                                     ("ERA5_trained_LSTM", "F_EE", "F_EG"),
                                     ("GFS_trained_LSTM", "F_GE", "F_GG")]}
    return {"points": len(frame), "origins": int(frame.issue_time.nunique()),
            "unique_target_hours": int(frame.target_time.nunique()), "metrics": metrics,
            "mse_source_decomposition": mse, "gain_gap_decomposition": gain_decomposition(rates),
            "denominator_decomposition": denominator_decomposition(
                *(metrics[c]["rmse_kw"] for c in ("B_E", "F_EE", "B_G", "F_GG")))}


def weather_differences(frame):
    records = []
    for name, unit in zip(NAMES, ["m/s", "kg/m^3", "dimensionless", "kW"], strict=True):
        if f"era5_{name}" in frame and f"gfs_{name}" in frame:
            s = score(frame[f"era5_{name}"], frame[f"gfs_{name}"])
            records.append({"feature": name, "unit": unit, "gfs_minus_era5_rmse": s["rmse_kw"],
                            "gfs_minus_era5_mae": s["mae_kw"], "gfs_minus_era5_bias": s["bias_kw"]})
    return records


def attribute(frame, output):
    frame = frame.copy()
    for column in ("issue_time", "target_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if frame.duplicated(KEYS).any() or frame[KEYS].isna().any().any():
        raise ValueError("Forecast keys must be unique and complete.")
    delta = (frame.target_time - frame.issue_time).dt.total_seconds() / 3600
    if not np.array_equal(delta, frame.lead_hours):
        raise ValueError("Lead times do not match issue/target timestamps.")
    if not np.isfinite(frame[["target_power_kw", *CELLS]].to_numpy()).all():
        raise ValueError("All six cells need finite predictions on the same rows.")
    groups = [("overall", "all", frame)]
    for kind, key in [("season", "segment_id"), ("day", "issue_time"), ("lead", "lead_hours")]:
        groups.extend((kind, str(name), group) for name, group in frame.groupby(key, sort=True))
    groups.extend(("leave_one_season_out", str(name), frame.loc[frame.segment_id.ne(name)])
                  for name in sorted(frame.segment_id.unique()) if frame.segment_id.nunique() > 1)
    grouped, metric_rows, gain_rows, mse_rows, weather_rows = {}, [], [], [], []
    for kind, name, group in groups:
        result = analyze_frame(group)
        grouped.setdefault(kind, {})[name] = result
        keys = {"scope": kind, "group": name, "points": len(group)}
        metric_rows.extend({**keys, "cell": cell, **metric} for cell, metric in result["metrics"].items())
        gain_rows.append({**keys, **result["gain_gap_decomposition"],
                          **{f"denominator_view_{k}": v for k, v in result["denominator_decomposition"].items()}})
        mse_rows.extend({**keys, "method": method, **value} for method, value in result["mse_source_decomposition"].items())
        weather_rows.extend({**keys, **value} for value in weather_differences(group))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "predictions.csv", index=False)
    for name, rows in [("metrics_by_scope", metric_rows), ("attribution_by_scope", gain_rows),
                       ("mse_by_scope", mse_rows), ("weather_difference_by_scope", weather_rows)]:
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    result = {"overall": grouped["overall"]["all"], "weather_differences": weather_differences(frame)}
    for name, value in [("summary", result), ("grouped_results", grouped)]:
        (output / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def evaluate(data_path, model_root, output):
    with np.load(data_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    era5, gfs, truth = data["era5_x"], data["gfs_x"], data["y"]
    if era5.shape != gfs.shape or era5.shape[:2] != truth.shape:
        raise ValueError("Weather inputs and reference power must cover the same windows.")
    n = len(truth)
    frame = pd.DataFrame({"segment_id": np.repeat(data["segment_ids"], 24),
                          "issue_time": np.repeat(data["issue_times"], 24),
                          "target_time": data["target_times"].ravel(),
                          "lead_hours": np.tile(np.arange(1, 25), n),
                          "target_power_kw": truth.ravel(), "B_E": era5[:, :, 3].ravel(),
                          "B_G": gfs[:, :, 3].astype(float).ravel()})
    for source, values in [("era5", era5), ("gfs", gfs)]:
        for index, name in enumerate(NAMES):
            frame[f"{source}_{name}"] = values[:, :, index].astype(float).ravel()
    for cell, kind, values in [("F_EE", "era5", era5), ("F_EG", "era5", gfs),
                               ("F_GE", "gfs", era5), ("F_GG", "gfs", gfs)]:
        frame[cell] = predict(Path(model_root) / kind, values, kind).astype(float).ravel()
    return attribute(frame, output)
