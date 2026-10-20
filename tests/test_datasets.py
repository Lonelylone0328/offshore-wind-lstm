import numpy as np
import pandas as pd
import pytest

from windpower.datasets import build_dataset
from windpower.physics import NAMES


def write_inputs(tmp_path, times):
    features = pd.DataFrame({"time": times, **{name: np.arange(len(times)) + 1.0 for name in NAMES}})
    labels = pd.DataFrame({"time": times, "target_power_kw": np.arange(len(times)) + 100.0})
    feature_path, label_path = tmp_path / "features.csv", tmp_path / "labels.csv"
    features.to_csv(feature_path, index=False)
    labels.to_csv(label_path, index=False)
    return features, labels, feature_path, label_path


def test_era5_windows_keep_gaps_and_seasons_separate(tmp_path):
    times = pd.date_range("2023-01-01", periods=25, freq="h").append([
        pd.date_range("2023-01-04", periods=24, freq="h"),
        pd.date_range("2023-07-01", periods=24, freq="h"),
    ])
    _, labels, feature_path, label_path = write_inputs(tmp_path, times)
    labels.sample(frac=1, random_state=1).to_csv(label_path, index=False)
    output = tmp_path / "windows.npz"
    stats = build_dataset("era5", feature_path, label_path, output)
    assert stats["windows"] == 4
    assert stats["train_windows"] == 3
    assert stats["validation_windows"] == 1
    with np.load(output) as data:
        winter = np.flatnonzero(data["segment_ids"] == "winter")
        np.testing.assert_array_equal(data["y"][winter[0]], np.arange(24) + 100)
        assert data["target_times"].shape == (4, 24)
        assert np.all(np.diff(data["target_times"], axis=1) == np.timedelta64(1, "h"))
        assert np.all(data["issue_times"] + np.timedelta64(1, "h") == data["target_times"][:, 0])


def test_gfs_keeps_supplied_curve_and_hour_identity(tmp_path):
    times = pd.date_range("2023-07-01", periods=24, freq="h", tz="UTC")
    features, _, feature_path, label_path = write_inputs(tmp_path, times)
    features = features.rename(columns={"time": "target_time"})
    features["issue_time"] = times[0] - pd.Timedelta(hours=1)
    features["cycle_time"] = times[0] - pd.Timedelta(hours=12)
    features["available_time"] = times[0] - pd.Timedelta(hours=8)
    features["forecast_hour"] = np.arange(12, 36)
    features[NAMES[3]] = 1234.56789
    features.sample(frac=1, random_state=2).to_csv(feature_path, index=False)
    output = tmp_path / "windows.npz"
    stats = build_dataset("gfs", feature_path, label_path, output)
    assert stats["train_windows"] == 1 and stats["validation_windows"] == 0
    with np.load(output) as data:
        assert data["x"].dtype == np.float32 and data["y"].dtype == np.float64
        np.testing.assert_array_equal(data["x"][0, :, 3], np.full(24, 1234.56789, dtype=np.float32))
        np.testing.assert_array_equal(data["y"][0], np.arange(24) + 100)
    features["available_time"] = times[0]
    features.to_csv(feature_path, index=False)
    with pytest.raises(ValueError, match="source timing"):
        build_dataset("gfs", feature_path, label_path, output)


def test_missing_weather_is_not_silently_paired(tmp_path):
    times = pd.date_range("2023-01-01", periods=24, freq="h")
    features, _, feature_path, label_path = write_inputs(tmp_path, times)
    features.iloc[:-1].to_csv(feature_path, index=False)
    with pytest.raises(ValueError, match="finite weather"):
        build_dataset("era5", feature_path, label_path, tmp_path / "windows.npz")
