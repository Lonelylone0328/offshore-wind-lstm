"""Build training windows from physical features and hourly power labels."""

from pathlib import Path

import numpy as np
import pandas as pd

from .physics import NAMES

SEASONS = np.array(["winter", "spring", "summer", "autumn"])


def build_dataset(kind, features_path, labels_path, output,
                  train_segments=("winter", "spring"), validation_segments=("summer",)):
    """Save the selected ERA5 rolling or GFS daily training-window format."""
    if kind not in ("era5", "gfs"):
        raise ValueError("Dataset kind must be 'era5' or 'gfs'.")
    features = pd.read_csv(features_path, float_precision="round_trip")
    labels = pd.read_csv(labels_path, float_precision="round_trip")
    labels["time"] = pd.to_datetime(labels.time, utc=True)
    if labels.time.duplicated().any():
        raise ValueError("Power label timestamps must be unique.")
    if "segment_id" not in labels:
        labels["segment_id"] = SEASONS[(labels.time.dt.month % 12) // 3]
    labels = labels.set_index("time").sort_index()
    rows = []
    if kind == "era5":
        features["time"] = pd.to_datetime(features.time, utc=True)
        if features.time.duplicated().any():
            raise ValueError("ERA5 feature timestamps must be unique.")
        aligned = features.set_index("time").reindex(labels.index)[NAMES].join(
            labels[["target_power_kw", "segment_id"]])
        discontinuity = (aligned.index.to_series().diff() != pd.Timedelta(hours=1))
        discontinuity |= aligned.segment_id.ne(aligned.segment_id.shift())
        for _, run in aligned.groupby(discontinuity.cumsum(), sort=False):
            for start in range(len(run) - 23):
                window = run.iloc[start:start + 24]
                rows.append((window.index[0] - pd.Timedelta(hours=1), window))
    else:
        for name in ("issue_time", "target_time"):
            features[name] = pd.to_datetime(features[name], utc=True)
        for issue, group in features.groupby("issue_time", sort=True):
            group = group.sort_values("target_time").set_index("target_time")
            expected = pd.date_range(issue + pd.Timedelta(hours=1), periods=24, freq="h")
            if not group.index.equals(expected):
                raise ValueError("Each GFS origin needs consecutive t+1..24 target hours.")
            available = pd.to_datetime(group.available_time, utc=True)
            cycles = pd.to_datetime(group.cycle_time, utc=True)
            valid_times = cycles + pd.to_timedelta(group.forecast_hour, unit="h")
            if (available.isna().any() or cycles.isna().any() or (available > issue).any()
                    or (cycles > issue - pd.Timedelta(hours=6)).any()
                    or not valid_times.eq(group.index).all()):
                raise ValueError("GFS source timing must match the available forecast protocol.")
            window = group[NAMES].join(labels[["target_power_kw", "segment_id"]])
            if window.segment_id.nunique(dropna=False) != 1:
                raise ValueError("A GFS training window must belong to one labeled segment.")
            rows.append((issue, window))
    if not rows:
        raise ValueError("No complete 24-hour training windows were found.")
    rows.sort(key=lambda item: (str(item[1].segment_id.iloc[0]), item[0]))
    x = np.stack([window[NAMES].to_numpy(dtype=np.float32 if kind == "gfs" else float)
                  for _, window in rows])
    y = np.stack([window.target_power_kw.to_numpy(dtype=float) for _, window in rows])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Every target hour needs finite weather features and a power label.")
    issues = np.array([issue.tz_localize(None).to_datetime64() for issue, _ in rows])
    times = np.stack([window.index.tz_localize(None).to_numpy() for _, window in rows])
    segments = np.array([str(window.segment_id.iloc[0]) for _, window in rows])
    training = np.isin(segments, train_segments) if kind == "era5" else np.ones(len(rows), dtype=bool)
    validation = np.isin(segments, validation_segments) if kind == "era5" else np.zeros(len(rows), dtype=bool)
    if np.any(training & validation):
        raise ValueError("Training and validation segments must be separate.")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=x, y=y, target_times=times, issue_times=issues,
                        segment_ids=segments, train_mask=training, validation_mask=validation)
    return {
        "output": str(output), "windows": len(rows),
        "train_windows": int(training.sum()), "validation_windows": int(validation.sum()),
        "unique_target_hours": int(np.unique(times).size),
    }
