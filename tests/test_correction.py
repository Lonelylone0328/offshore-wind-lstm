from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from windpower.correction import build_targets, inputs, load_dataset, predict, read_csv
from windpower.correction_training import split_days, train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def data():
    torch.set_num_threads(1)
    return load_dataset(ROOT / "data/native_test.npz")


@pytest.mark.parametrize("route,column", [
    ("weather", "weather_lstm_power_kw"),
    ("power", "direct_power_lstm_kw"),
])
def test_frozen_annual_predictions(data, route, column):
    expected = pd.read_csv(ROOT / f"data/expected_{route}.csv")
    np.testing.assert_array_equal(
        pd.to_datetime(expected.target_time, utc=True).dt.tz_localize(None).to_numpy(),
        data["valid_times"][:, 24:].ravel())
    values = predict(ROOT / "models" / route, data, route, ROOT / "models/era5")
    np.testing.assert_allclose(values.ravel(), expected[column], rtol=0, atol=1e-5)


def test_unavailable_weather_is_rejected(data):
    data["available_times"][0, -1] = data["issue_times"][0]
    with pytest.raises(ValueError, match="deadline"):
        inputs(data)


def test_target_shift_is_rejected(data):
    data["valid_times"][0, -1] += np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="24 history"):
        inputs(data)


def test_era5_targets_reproduce_training_supervision(tmp_path):
    path = tmp_path / "targets.npz"
    result = build_targets(ROOT / "data/native_train.npz", ROOT / "data/era5_2023.csv", path)
    actual = load_dataset(path)
    expected = load_dataset(ROOT / "data/native_targets.npz")
    assert result["target_hours"] == 365 * 24
    np.testing.assert_array_equal(actual["issue_times"], expected["issue_times"])
    np.testing.assert_array_equal(actual["state"], expected["state"])


def test_csv_matches_native_features(data, tmp_path):
    points = data["points_lat_lon"]
    frame = pd.DataFrame(data["raw_values"][0].reshape(-1, 7), columns=data["raw_columns"])
    frame["issue_time"] = data["issue_times"][0]
    for name, key in [("valid_time", "valid_times"), ("cycle_time", "cycle_times"),
                      ("available_time", "available_times")]:
        frame[name] = np.repeat(data[key][0], len(points))
    frame["latitude"] = np.tile(points[:, 0], 48)
    frame["longitude"] = np.tile(points[:, 1], 48)
    path = tmp_path / "weather.csv"
    frame.to_csv(path, index=False)
    parsed = read_csv(path)
    x, state, _ = inputs(data)
    csv_x, csv_state, _ = inputs(parsed)
    np.testing.assert_allclose(csv_x, x[:1], rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(csv_state, state[:1], rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("route", ["weather", "power"])
def test_training_produces_reloadable_ensemble(route, tmp_path, data):
    annual = load_dataset(ROOT / "data/native_train.npz")
    train_mask, valid = split_days(annual)
    assert (train_mask.sum(), valid.sum(), (~(train_mask | valid)).sum()) == (293, 60, 12)
    folder = tmp_path / route
    result = train(route, ROOT / "data/native_train.npz", ROOT / "data/native_targets.npz",
                   folder, ROOT / "models/era5", max_epochs=1)
    assert len(result["checkpoints"]) == 3
    prediction = predict(folder, data, route, ROOT / "models/era5")
    assert prediction.shape == (28, 24)
    assert np.isfinite(prediction).all()
