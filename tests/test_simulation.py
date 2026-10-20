import struct

import numpy as np
import pandas as pd
import pytest

from windpower import acquisition
from windpower.acquisition import source_jobs
from windpower.simulation import SimulationCase, inspect_wind_field, summarize_output


def write_output(path, case, *, pitch=None, end=720.0):
    times = np.round(np.arange(0, end + 0.05, 0.1), 1)
    power = np.where(times <= case.warmup_s, 9000, 1200 if not case.parked else 0)
    pitch_values = np.full_like(times, 90 if case.parked else 0) if pitch is None else pitch
    rotor = np.full_like(times, 0 if case.parked else 12)
    with path.open("w") as stream:
        stream.write("Test fixture\nTime GenPwr BldPitch1 RotSpeed\n(s) (kW) (deg) (rpm)\n")
        np.savetxt(stream, np.column_stack([times, power, pitch_values, rotor]), fmt="%.6f")


def test_label_excludes_warmup(tmp_path):
    case = SimulationCase("test", "2024-01-01T00:00:00Z", 8)
    path = tmp_path / "output.out"
    write_output(path, case)
    stats = summarize_output(path, case)
    assert stats["target_power_kw"] == 1200
    assert stats["retained_samples"] == 6000
    assert stats["effective_seconds"] == 600


def test_truncated_or_unphysical_output_rejected(tmp_path):
    case = SimulationCase("test", "2024-01-01T00:00:00Z", 8)
    path = tmp_path / "output.out"
    write_output(path, case, end=719.9)
    with pytest.raises(ValueError, match="full simulation"):
        summarize_output(path, case)
    pitch = np.zeros(7201)
    pitch[100:] = 90
    write_output(path, case, pitch=pitch)
    with pytest.raises(ValueError, match="pitch"):
        summarize_output(path, case)


def test_parked_output(tmp_path):
    case = SimulationCase("test", "2024-01-01T00:00:00Z", 2)
    path = tmp_path / "output.out"
    write_output(path, case)
    assert summarize_output(path, case)["target_power_kw"] == 0


def write_wind(path, nt, base=5):
    header = struct.pack(
        "<h4i12fi", 7, 15, 15, 0, nt, 170 / 14, 170 / 14, 0.1, 8, 90, base,
        1, 0, 1, 0, 1, 0, 0,
    )
    with path.open("wb") as stream:
        stream.write(header)
        stream.truncate(70 + nt * 3 * 15 * 15 * 2)


def test_real_wind_geometry_and_taylor_coverage(tmp_path):
    case = SimulationCase("test", "2024-01-01T00:00:00Z", 8)
    path = tmp_path / "wind.bts"
    write_wind(path, 7500)
    assert inspect_wind_field(path, case)["base_height_m"] == 5
    write_wind(path, 7201)
    with pytest.raises(ValueError, match="Taylor"):
        inspect_wind_field(path, case)
    write_wind(path, 7500, base=0.0001)
    with pytest.raises(ValueError, match="geometry"):
        inspect_wind_field(path, case)


def test_gfs_interpolation_and_native_history():
    issue = pd.Timestamp("2024-01-14T23:00:00Z")
    times, jobs = source_jobs(issue, pd.Timestamp("2024-01-14T12:00:00Z"), 0, 0.5)
    assert len(times) == 24
    assert jobs[1][0][1] == 12 and jobs[1][1][1] == 15
    assert jobs[1][2] == pytest.approx(1 / 3)
    assert jobs[-1][0][1] == 33 and jobs[-1][1][1] == 36
    cycle = pd.Timestamp("2024-01-14T18:00:00Z")
    times, jobs = source_jobs(issue, cycle, 24, 0.25)
    assert len(times) == 48
    assert times[0] == issue - pd.Timedelta(hours=23)
    assert times[-1] == issue + pd.Timedelta(hours=24)
    assert jobs[0][0] == (cycle - pd.Timedelta(days=1), 6)
    assert jobs[18][0] == (cycle, 0)
    assert jobs[-1][0] == (cycle, 29)


def test_gfs_download_uses_one_available_cycle_and_interpolates(tmp_path, monkeypatch):
    pytest.importorskip("requests")
    issue = pd.Timestamp("2024-01-14T23:00:00Z")

    def fake_object(job, resolution, sites, cache):
        cycle, hour = job
        available = issue + pd.Timedelta(minutes=1) if cycle.hour == 12 else issue - pd.Timedelta(hours=1)
        return {"values": np.full((len(sites), 7), hour ** 2), "available_time": available,
                "key": f"{cycle:%Y%m%d%H}/f{hour:03d}"}

    monkeypatch.setattr(acquisition, "download_gfs_object", fake_object)
    output = tmp_path / "gfs.csv"
    frame = acquisition.download_gfs([issue], output, tmp_path / "cache", workers=1)
    assert len(frame) == 48
    assert frame.cycle_time.nunique() == 1
    assert frame.cycle_time.iloc[0] == pd.Timestamp("2024-01-14T06:00:00Z")
    assert (frame.available_time <= issue).all()
    assert frame.loc[(frame.lead_hours == 2) & (frame.site_id == "sea"), "u10"].item() == pytest.approx(363)
    assert output.with_suffix(".source.json").is_file()


def test_native_download_exports_correction_input(tmp_path, monkeypatch):
    pytest.importorskip("requests")
    issue = pd.Timestamp("2024-01-14T23:00:00Z")
    sites = [
        (f"{name}_{dy}_{dx}", lat + dy, lon + dx)
        for name, lat, lon in acquisition.SITES
        for dy in (-0.25, 0, 0.25) for dx in (-0.25, 0, 0.25)
    ]

    def fake_object(job, resolution, sites, cache):
        return {"values": np.zeros((18, 7)), "available_time": issue - pd.Timedelta(hours=1),
                "key": f"{job[0]:%Y%m%d%H}/f{job[1]:03d}"}

    monkeypatch.setattr(acquisition, "download_gfs_object", fake_object)
    archive = tmp_path / "native.npz"
    acquisition.download_gfs(
        [issue], tmp_path / "native.csv", tmp_path / "cache", resolution=0.25,
        sites=sites, history_hours=24, workers=1, npz_output=archive,
    )
    with np.load(archive) as data:
        assert data["raw_values"].shape == (1, 48, 18, 7)
        assert np.allclose(data["points_lat_lon"][[4, 13]], [[33, 122], [33, 120]])
        assert data["valid_times"][0, 24] == np.datetime64("2024-01-15T00:00:00")
        assert data["available_times"].dtype == np.dtype("datetime64[ns]")
