"""Native hourly GFS weather and power correction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .models import ConditionalLSTM
from .physics import empirical_power_curve, moist_air_properties

RAW_COLUMNS = ["u10", "v10", "u100", "v100", "t2m", "d2m", "sp"]
STATE_COLUMNS = ["hub_u_mps", "hub_v_mps", "air_density_kgm3", "shear_exponent"]
STATE_SCALES = np.array([5., 5., .05, .1])
POINTS = np.array([(33. + dy, lon + dx) for lon in (122., 120.)
                   for dy in (-.25, 0., .25) for dx in (-.25, 0., .25)])


def state_from_raw(raw):
    u10, v10, u100, v100, temperature, dewpoint, pressure = np.moveaxis(
        np.asarray(raw, dtype=float), -1, 0)
    low, high = np.hypot(u10, v10), np.hypot(u100, v100)
    safe = (low >= .5) & (high >= .5)
    alpha = np.log(np.where(safe, high, 1.) / np.where(safe, low, 1.)) / np.log(10.)
    alpha = np.where(safe & (alpha >= -.2) & (alpha <= .5), alpha, .14)
    _, density = moist_air_properties(temperature, dewpoint, pressure)
    return np.stack([u100 * .9 ** alpha, v100 * .9 ** alpha, density, alpha], axis=-1)


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def build_targets(data_path, era5_csv, output):
    data = load_dataset(data_path)
    frame = pd.read_csv(era5_csv)
    frame = frame.loc[frame.site_id.eq("sea")].copy()
    frame["time"] = pd.to_datetime(frame.time, utc=True).dt.tz_localize(None)
    if frame.time.duplicated().any():
        raise ValueError("ERA5 sea-site timestamps must be unique.")
    times = data["valid_times"][:, 24:]
    raw = frame.set_index("time").reindex(times.ravel())[RAW_COLUMNS].to_numpy()
    if not np.isfinite(raw).all():
        raise ValueError("ERA5 must cover every training target hour.")
    state = state_from_raw(raw).reshape(*times.shape, 4)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, state=state, issue_times=data["issue_times"])
    return {"output": str(output), "origins": len(state), "target_hours": int(times.size)}


def read_csv(path):
    frame = pd.read_csv(path)
    for name in ("issue_time", "valid_time", "cycle_time", "available_time"):
        frame[name] = pd.to_datetime(frame[name], utc=True).dt.tz_localize(None)
    issues = np.array(sorted(frame.issue_time.unique()), dtype="datetime64[ns]")
    values, valid_rows, cycles, availability = [], [], [], []
    for issue in issues:
        times = pd.date_range(pd.Timestamp(issue) - pd.Timedelta(hours=23), periods=48, freq="h")
        keys = pd.MultiIndex.from_tuples(
            [(time, lat, lon) for time in times for lat, lon in POINTS],
            names=["valid_time", "latitude", "longitude"])
        aligned = frame.loc[frame.issue_time == issue].set_index(keys.names).reindex(keys)
        init = aligned.cycle_time.to_numpy().reshape(48, 18)
        ready = aligned.available_time.to_numpy().reshape(48, 18)
        if not np.all(init == init[:, :1]) or not np.all(ready == ready[:, :1]):
            raise ValueError("Each valid hour must use one forecast cycle and availability time.")
        values.append(aligned[RAW_COLUMNS].to_numpy().reshape(48, 18, 7))
        valid_rows.append(times.to_numpy())
        cycles.append(init[:, 0])
        availability.append(ready[:, 0])
    return {"raw_values": np.array(values), "raw_columns": np.array(RAW_COLUMNS),
            "points_lat_lon": POINTS.copy(), "issue_times": issues,
            "valid_times": np.array(valid_rows), "cycle_times": np.array(cycles),
            "available_times": np.array(availability)}


def inputs(data):
    raw = data["raw_values"]
    issues = data["issue_times"].astype("datetime64[ns]")
    expected = issues[:, None] + np.arange(-23, 25)[None, :] * np.timedelta64(1, "h")
    if raw.shape != (len(issues), 48, 18, 7) or not np.isfinite(raw).all():
        raise ValueError("Each issue needs 48 complete hours at 18 points.")
    if data["raw_columns"].tolist() != RAW_COLUMNS or not np.array_equal(data["points_lat_lon"], POINTS):
        raise ValueError("Weather fields or neighborhood points are out of order.")
    if (not np.array_equal(data["valid_times"], expected)
            or len(np.unique(issues)) != len(issues)
            or np.any(issues != issues.astype("datetime64[D]") + np.timedelta64(23, "h"))):
        raise ValueError("Use unique daily 23 UTC issues with 24 history and 24 forecast hours.")
    cycle, available = data["cycle_times"], data["available_times"]
    if (np.isnat(cycle).any() or np.isnat(available).any()
            or np.any(available > issues[:, None] - np.timedelta64(15, "m"))
            or np.any(cycle > available) or np.any(cycle > expected)
            or np.any(expected - cycle > np.timedelta64(120, "h"))):
        raise ValueError("Forecast timing exceeds the issue deadline or native hourly range.")
    state = state_from_raw(raw[:, :, 4])
    times = pd.DatetimeIndex(data["valid_times"].ravel())
    time_x = np.stack([
        np.sin(2 * np.pi * times.hour / 24), np.cos(2 * np.pi * times.hour / 24),
        np.sin(2 * np.pi * (times.dayofyear - 1) / 365.25),
        np.cos(2 * np.pi * (times.dayofyear - 1) / 365.25),
    ], axis=-1).reshape(len(raw), 48, 4)
    native_lead = ((data["valid_times"] - cycle) / np.timedelta64(1, "h"))[..., None] / 24.
    relative_lead = np.broadcast_to(np.arange(-23, 25)[None, :, None] / 24., (len(raw), 48, 1))
    speed = np.hypot(state[..., 0], state[..., 1])
    power = empirical_power_curve(speed)[..., None] / 5000.
    direction_defined = speed >= .5
    physical = np.stack([
        speed,
        np.where(direction_defined, -state[..., 0] / np.maximum(speed, .5), 0.),
        np.where(direction_defined, -state[..., 1] / np.maximum(speed, .5), 0.),
        direction_defined.astype(float),
        raw[:, :, 13, 4] - raw[:, :, 4, 4],
        raw[:, :, 13, 6] - raw[:, :, 4, 6],
        (raw[:, :, 5, 2] - raw[:, :, 3, 2]) / .5,
        (raw[:, :, 7, 3] - raw[:, :, 1, 3]) / .5,
    ], axis=-1)
    x = np.concatenate([raw.reshape(len(raw), 48, -1), state, power,
                        physical, time_x, native_lead, relative_lead], axis=-1)
    columns = [f"point{point}_{field}" for point in range(18) for field in RAW_COLUMNS]
    columns += STATE_COLUMNS + [
        "curve_fraction", "hub_speed_mps", "wind_from_sin", "wind_from_cos", "direction_defined",
        "land_minus_sea_temperature_k", "land_minus_sea_pressure_pa",
        "sea_u100_east_gradient_mps_per_degree", "sea_v100_north_gradient_mps_per_degree",
        "hour_sin", "hour_cos", "year_sin", "year_cos", "native_lead_days", "relative_lead_days",
    ]
    return x, state[:, 24:], columns


def torch_curve(speed):
    power = torch.clamp((speed.pow(3) - 27.) / (11.4 ** 3 - 27.), 0., 1.)
    return torch.where((speed < 3.) | (speed >= 25.), torch.zeros_like(power), power)


class FrozenPower(nn.Module):
    def __init__(self, teacher_dir):
        super().__init__()
        folder = Path(teacher_dir)
        checkpoints = json.loads((folder / "model.json").read_text())["checkpoints"]
        models, means, scales = [], [], []
        for name in checkpoints:
            saved = torch.load(folder / name, map_location="cpu", weights_only=True)
            config = saved["model_config"]
            model = ConditionalLSTM(config["hidden_size"], config["residual"])
            model.load_state_dict(saved["model_state_dict"])
            models.append(model.eval().requires_grad_(False))
            means.append(saved["scaler_mean"])
            scales.append(saved["scaler_std"])
        self.models = nn.ModuleList(models)
        self.register_buffer("means", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.eval()

    def forward(self, state):
        speed = torch.sqrt(state[..., 0].square() + state[..., 1].square() + 1e-12)
        curve = torch_curve(speed)
        weather = torch.stack([speed, state[..., 2], state[..., 3], curve], dim=-1)
        return torch.stack([model((weather - mean) / scale, curve) for model, mean, scale
                            in zip(self.models, self.means, self.scales, strict=True)]).mean(0) * 5000.


class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=16, bidirectional=True):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh())
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(hidden_size * (2 if bidirectional else 1), 4)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer("state_scales", torch.tensor(STATE_SCALES, dtype=torch.float32))

    def forward(self, x, base):
        hidden, _ = self.encoder(self.project(x))
        raw = base + self.head(hidden[:, -24:]) * self.state_scales
        return torch.stack([raw[..., 0], raw[..., 1],
                            raw[..., 2].clamp(.8, 1.5), raw[..., 3].clamp(-.2, .5)], dim=-1)


class PowerResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=16, bidirectional=True):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh())
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=bidirectional)
        self.head = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, base_power, *, bounded=True):
        hidden, _ = self.encoder(self.project(x))
        prediction = base_power + 5000. * self.head(hidden[:, -24:]).squeeze(-1)
        return prediction.clamp(0., 5000.) if bounded else prediction


def predict(model_dir, data, route, teacher_dir):
    folder = Path(model_dir)
    record = json.loads((folder / "model.json").read_text())
    if route not in {"weather", "power"} or record["route"] != route:
        raise ValueError("Choose the matching weather or power model.")
    raw, base_state, _ = inputs(data)
    teacher = FrozenPower(teacher_dir)
    with torch.no_grad():
        base_state = torch.tensor(base_state, dtype=torch.float32)
        base = teacher(base_state) if route == "power" else base_state
        outputs = []
        for name in record["checkpoints"]:
            saved = torch.load(folder / name, map_location="cpu", weights_only=True)
            config = saved["config"]
            cls = PowerResidualLSTM if route == "power" else WeatherLSTM
            model = cls(raw.shape[-1], config["hidden_size"], config["bidirectional"])
            model.load_state_dict(saved["model_state_dict"])
            model.eval()
            x = (raw - np.array(saved["scaler_mean"])) / np.array(saved["scaler_std"])
            outputs.append(model(torch.tensor(x, dtype=torch.float32), base).numpy())
        if route == "power":
            return np.mean(outputs, axis=0, dtype=np.float64)
        return teacher(torch.tensor(np.mean(outputs, axis=0), dtype=torch.float32)).numpy()
