"""Physical features for the 90 m hub-height, 5 MW reference turbine."""

import numpy as np
import pandas as pd

NAMES = [
    "hub_wind_speed_mps",
    "air_density_kgm3",
    "shear_exponent",
    "theoretical_power_kw",
]


def moist_air_properties(temperature_k, dewpoint_k, pressure_pa):
    """Return relative humidity (%) and moist-air density (kg/m³)."""
    temperature = np.asarray(temperature_k, dtype=float)
    dewpoint = np.asarray(dewpoint_k, dtype=float)
    pressure = np.asarray(pressure_pa, dtype=float)
    vapor = 611.21 * np.exp(17.502 * (dewpoint - 273.16) / (dewpoint - 32.19))
    saturated = 611.21 * np.exp(17.502 * (temperature - 273.16) / (temperature - 32.19))
    humidity = 100.0 * vapor / saturated
    density = (pressure - vapor) / (287.05 * temperature) + vapor / (461.5 * temperature)
    return humidity, density


def empirical_power_curve(wind_speed_mps):
    """Piecewise cubic reference curve in kW."""
    speed = np.asarray(wind_speed_mps, dtype=float)
    valid = np.isfinite(speed) & (speed >= 0)
    fraction = (speed**3 - 3.0**3) / (11.4**3 - 3.0**3)
    power = 5000.0 * np.clip(fraction, 0, 1)
    power = np.where((speed < 3.0) | (speed >= 25.0), 0.0, power)
    return np.where(valid, power, np.nan)


def build_features(frame):
    """Convert hourly sea-site u/v winds, temperature, dewpoint and pressure."""
    sea = frame.loc[frame.site_id.eq("sea")].copy() if "site_id" in frame else frame.copy()
    sea["time"] = pd.to_datetime(sea.time, utc=True)
    sea = sea.sort_values("time").reset_index(drop=True)
    if sea.time.duplicated().any():
        raise ValueError("Sea-site timestamps must be unique.")
    low = np.hypot(sea.u10, sea.v10)
    high = np.hypot(sea.u100, sea.v100)
    safe = (low >= 0.5) & (high >= 0.5)
    alpha = np.log(high.where(safe) / low.where(safe)) / np.log(10.0)
    alpha = alpha.where(alpha.between(-0.2, 0.5), 0.14)
    speed = high * 0.9**alpha
    _, density = moist_air_properties(sea.t2m, sea.d2m, sea.sp)
    return pd.DataFrame({
        "time": sea.time,
        NAMES[0]: speed,
        NAMES[1]: density,
        NAMES[2]: alpha,
        NAMES[3]: empirical_power_curve(speed),
    })
