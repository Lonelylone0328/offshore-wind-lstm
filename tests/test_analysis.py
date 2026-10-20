import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from windpower.analysis import (
    CELLS,
    KEYS,
    attribute,
    denominator_decomposition,
    evaluate,
    gain_decomposition,
    improvement,
    mse_decomposition,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("gfs,expected_mse,interaction", [
    ([1., -1.], 1., -4.),
    ([3., -3.], 9., 4.),
])
def test_mse_retains_signed_interaction(gfs, expected_mse, interaction):
    result = mse_decomposition([0., 0.], [2., -2.], gfs)
    assert result["era5_mse_kw2"] == 4.
    assert result["gfs_mse_kw2"] == expected_mse
    assert result["source_shift_mse_kw2"] == 1.
    assert result["interaction_kw2"] == interaction
    assert result["closure_residual_kw2"] == 0.


def test_gain_decomposition_preserves_negative_contribution():
    result = gain_decomposition({"F_EE": 40., "F_EG": 2., "F_GE": -60., "F_GG": 4.})
    assert result["total_gap_pp"] == 36.
    assert result["weather_source_pp"] == -13.
    assert result["frozen_model_pp"] == 49.
    assert result["weather_share_of_gap_percent"] < 0.
    assert result["model_share_of_gap_percent"] > 100.
    assert result["closure_residual_pp"] == 0.


@pytest.mark.parametrize("rmse,savings,denominator,gap", [
    ((200., 150., 1000., 950.), 0., 20., 20.),
    ((200., 100., 200., 150.), 25., 0., 25.),
])
def test_gain_denominator_known_cases(rmse, savings, denominator, gap):
    result = denominator_decomposition(*rmse)
    assert result["absolute_saving_change_pp"] == pytest.approx(savings)
    assert result["baseline_denominator_change_pp"] == pytest.approx(denominator)
    assert result["total_gap_pp"] == pytest.approx(gap)
    assert result["closure_residual_pp"] == pytest.approx(0.)


def test_zero_baseline_has_no_percentage_improvement():
    assert improvement(10., 0.) is None
    assert not denominator_decomposition(0., 0., 100., 90.)["defined"]
    assert not gain_decomposition({"F_EE": None, "F_EG": 2., "F_GE": None, "F_GG": 4.})["defined"]


def _compare_nested(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, dict):
            _compare_nested(actual[key], value)
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            np.testing.assert_allclose(actual[key], value, rtol=2e-12, atol=1e-8, err_msg=key)
        else:
            assert actual[key] == value


def test_all_six_cells_reproduce_reference_and_61_scopes(tmp_path):
    torch.set_num_threads(1)
    actual = evaluate(ROOT / "data/benchmark.npz", ROOT / "models", tmp_path)
    expected = json.loads((ROOT / "results/reference_summary.json").read_text())
    _compare_nested(actual["overall"], expected["overall"])

    calculated = pd.read_csv(tmp_path / "predictions.csv", float_precision="round_trip")
    reference = pd.read_csv(ROOT / "results/reference_predictions.csv", float_precision="round_trip")
    for frame in (calculated, reference):
        for key in ("issue_time", "target_time"):
            frame[key] = pd.to_datetime(frame[key], utc=True)
    pd.testing.assert_frame_equal(calculated[KEYS], reference[KEYS])
    columns = ["target_power_kw", *CELLS]
    np.testing.assert_allclose(calculated[columns], reference[columns], rtol=0, atol=1e-8)

    grouped = json.loads((tmp_path / "grouped_results.json").read_text())
    assert {key: len(value) for key, value in grouped.items()} == {
        "overall": 1, "season": 4, "day": 28, "lead": 24, "leave_one_season_out": 4,
    }
    assert len(pd.read_csv(tmp_path / "metrics_by_scope.csv")) == 61 * 6
    assert len(pd.read_csv(tmp_path / "mse_by_scope.csv")) == 61 * 3
    assert len(pd.read_csv(tmp_path / "attribution_by_scope.csv")) == 61
    for groups in grouped.values():
        for result in groups.values():
            assert abs(result["gain_gap_decomposition"]["closure_residual_pp"]) < 1e-9
            for method in result["mse_source_decomposition"].values():
                assert abs(method["closure_residual_kw2"]) < 1e-8


def test_target_and_lead_misalignment_is_rejected(tmp_path):
    frame = pd.read_csv(ROOT / "results/reference_predictions.csv")
    frame.loc[0, "lead_hours"] = 2
    with pytest.raises(ValueError, match="Lead times"):
        attribute(frame, tmp_path)
