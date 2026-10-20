from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from windpower.models import predict
from windpower.physics import NAMES, build_features, empirical_power_curve, moist_air_properties
from windpower.training import train

ROOT = Path(__file__).resolve().parents[1]


def test_curve_operating_boundaries():
    np.testing.assert_array_equal(
        empirical_power_curve([0, 3, 11.4, 24.9, 25]), [0, 0, 5000, 5000, 0]
    )
    assert 0 < empirical_power_curve(8) < 5000


def test_physical_features_and_fallback():
    humidity, density = moist_air_properties(288.15, 288.15, 101325)
    assert humidity == pytest.approx(100)
    assert 1.20 < density < 1.23
    raw = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=3, freq="h"),
        "u10": [8.0, 0.1, 1.0], "v10": [0.0, 0.0, 0.0],
        "u100": [10.0, 10.0, 10.0], "v100": [0.0, 0.0, 0.0],
        "t2m": 288.15, "d2m": 285.15, "sp": 101325,
    })
    features = build_features(raw)
    assert features.columns.tolist() == ["time", *NAMES]
    assert features[NAMES[2]].iloc[0] == pytest.approx(np.log(10 / 8) / np.log(10))
    np.testing.assert_array_equal(features[NAMES[2]].iloc[1:], [0.14, 0.14])
    assert np.isfinite(features[NAMES]).all().all()


@pytest.mark.parametrize("kind,source,expected", [
    ("era5", "era5_x", "expected_era5"),
    ("era5", "gfs_x", "expected_era5_on_gfs"),
    ("gfs", "era5_x", "expected_gfs_on_era5"),
    ("gfs", "gfs_x", "expected_gfs"),
])
def test_frozen_predictions(kind, source, expected):
    torch.set_num_threads(1)
    with np.load(ROOT / "data/benchmark.npz") as benchmark:
        result = predict(ROOT / "models" / kind, benchmark[source], kind)
        np.testing.assert_allclose(result, benchmark[expected], rtol=0, atol=0.001)


@pytest.mark.parametrize("kind,members", [("era5", 3), ("gfs", 5)])
def test_training_roundtrip_and_scaler_isolation(tmp_path, kind, members):
    hours = np.arange(25)
    base = np.column_stack([
        6 + hours / 24, 1.2 + hours / 2400,
        np.full(25, 0.14), empirical_power_curve(6 + hours / 24),
    ])
    x = np.stack([base[:24], base[1:], base[:24] * 1.5])
    origin = np.datetime64("2023-01-01T00")
    target_times = np.stack([
        origin + hours[:24].astype("timedelta64[h]"),
        origin + hours[1:].astype("timedelta64[h]"),
        origin + (100 + hours[:24]).astype("timedelta64[h]"),
    ])
    dataset = tmp_path / "training.npz"
    np.savez(dataset, x=x, y=x[:, :, 3] + 30, target_times=target_times,
             segment_ids=np.array(["winter", "winter", "summer"]),
             train_mask=[True, True, False], validation_mask=[False, False, True])
    output = tmp_path / "models"
    checkpoints = train(kind, dataset, output, max_epochs=1)
    assert len(checkpoints) == members
    result = predict(output, x, kind)
    assert result.shape == (3, 24) and np.isfinite(result).all()
    record = torch.load(checkpoints[0], weights_only=True)
    if kind == "era5":
        unique = base.copy()
        unique[:, 3] /= 5000
        np.testing.assert_allclose(record["scaler_mean"], unique.mean(axis=0))
    else:
        np.testing.assert_allclose(record["scaler"]["mean"], x[:2].reshape(-1, 4).mean(axis=0))
