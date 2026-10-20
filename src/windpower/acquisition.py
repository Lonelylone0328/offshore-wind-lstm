"""Point extraction from the ERA5 public mirror and NOAA GFS archive."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["u10", "v10", "u100", "v100", "t2m", "d2m", "sp"]
SITES = [("sea", 33.0, 122.0), ("land", 33.0, 120.0)]
GFS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
SPEC = {
    "u10": ("UGRD", "10 m above ground", 2, 2, 10),
    "v10": ("VGRD", "10 m above ground", 2, 3, 10),
    "u100": ("UGRD", "100 m above ground", 2, 2, 100),
    "v100": ("VGRD", "100 m above ground", 2, 3, 100),
    "t2m": ("TMP", "2 m above ground", 0, 0, 2),
    "d2m": ("DPT", "2 m above ground", 0, 6, 2),
    "sp": ("PRES", "surface", 3, 0, 0),
}
DECODE_LOCK = threading.Lock()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def download_era5(start, end, output, sites=SITES, snapshot=None):
    import icechunk
    import xarray as xr

    storage = icechunk.s3_storage(
        bucket="earthmover-icechunk-era5", prefix="icechunkV2", region="us-east-1",
        endpoint_url="https://s3.amazonaws.com", force_path_style=True, anonymous=True,
    )
    config = icechunk.RepositoryConfig.default()
    config.max_concurrent_requests = 8
    config.manifest = icechunk.ManifestConfig(
        preload=icechunk.ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0)
    )
    repository = icechunk.Repository.open(storage, config=config)
    session = repository.readonly_session(snapshot_id=snapshot) if snapshot else repository.readonly_session("main")
    remote = xr.open_zarr(session.store, group="single/temporal", consolidated=False, chunks=None)
    start = pd.Timestamp(start).tz_convert("UTC").tz_localize(None)
    end = pd.Timestamp(end).tz_convert("UTC").tz_localize(None)
    expected = pd.date_range(start, end, freq="h")
    rows = []
    for site_id, latitude, longitude in sites:
        point = remote.sel(latitude=latitude, longitude=longitude)
        arrays = {
            name: point[name].sel(valid_time=slice(start, end)).reset_coords(drop=True).load()
            for name in FIELDS
        }
        frame = xr.Dataset(arrays).to_dataframe().reset_index().rename(columns={"valid_time": "time"})
        if not pd.DatetimeIndex(frame.time).equals(expected) or not np.isfinite(frame[FIELDS]).all().all():
            raise ValueError(f"ERA5 hourly data are incomplete at {site_id}")
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["site_id"], frame["latitude"], frame["longitude"] = site_id, latitude, longitude
        frame["lsm"] = float(point.lsm.values)
        rows.append(frame)
    frame = pd.concat(rows).sort_values(["time", "site_id"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    write_json(output.with_suffix(".source.json"), {
        "source": "s3://earthmover-icechunk-era5/icechunkV2", "group": "single/temporal",
        "snapshot_id": str(session.snapshot_id), "dataset_doi": "10.24381/cds.adbb2d47",
        "retrieved_utc": datetime.now(UTC).isoformat(), "rows": len(frame),
        "units": {name: remote[name].attrs.get("units") for name in FIELDS},
    })
    remote.close()
    return frame


def gfs_key(cycle, hour, resolution):
    grid = "0p50" if resolution == 0.5 else "0p25"
    return f"gfs.{cycle:%Y%m%d}/{cycle:%H}/atmos/gfs.t{cycle:%H}z.pgrb2.{grid}.f{hour:03d}"


def fetch_cached(url, path, byte_range=None):
    import requests

    path = Path(path)
    metadata_path = path.with_suffix(path.suffix + ".json")
    if path.exists() and metadata_path.exists():
        return path.read_bytes(), json.loads(metadata_path.read_text())
    headers = {"Accept-Encoding": "identity"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    for attempt in range(4):
        try:
            with requests.get(url, headers=headers, timeout=(20, 180), stream=True) as response:
                response.raise_for_status()
                if byte_range and response.status_code != 206:
                    raise ValueError("GFS server did not honor the byte range")
                payload = response.content
                if byte_range and len(payload) != byte_range[1] - byte_range[0] + 1:
                    raise ValueError("Incomplete GFS byte range")
                metadata = {
                    "url": url, "last_modified": response.headers["Last-Modified"],
                    "etag": response.headers["ETag"], "byte_range": byte_range,
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                write_json(metadata_path, metadata)
                return payload, metadata
        except requests.RequestException as error:
            if attempt == 3 or getattr(error.response, "status_code", None) == 404:
                raise
            time.sleep(2 ** attempt)


def decode_field(payload, name, cycle, hour, resolution, sites):
    import eccodes

    with DECODE_LOCK:
        handle = eccodes.codes_new_from_message(payload)
        try:
            spec = SPEC[name]
            ni, nj = round(360 / resolution), round(180 / resolution) + 1
            expected = {
                "dataDate": int(cycle.strftime("%Y%m%d")), "dataTime": int(cycle.strftime("%H00")),
                "forecastTime": hour, "discipline": 0, "parameterCategory": spec[2],
                "parameterNumber": spec[3], "level": spec[4], "Ni": ni, "Nj": nj,
                "typeOfLevel": "surface" if name == "sp" else "heightAboveGround",
                "stepType": "instant", "gridType": "regular_ll",
                "latitudeOfFirstGridPointInDegrees": 90.0, "longitudeOfFirstGridPointInDegrees": 0.0,
                "iDirectionIncrementInDegrees": resolution, "jDirectionIncrementInDegrees": resolution,
                "iScansNegatively": 0, "jScansPositively": 0,
                "jPointsAreConsecutive": 0, "alternativeRowScanning": 0,
            }
            if any(eccodes.codes_get(handle, key) != value for key, value in expected.items()):
                raise ValueError(f"GFS field/time/grid mismatch: {name}, {cycle}, f{hour}")
            full = eccodes.codes_get_values(handle)
            values = []
            for _, latitude, longitude in sites:
                row, col = round((90 - latitude) / resolution), round((longitude % 360) / resolution)
                if not np.allclose([90 - row * resolution, col * resolution], [latitude, longitude % 360]):
                    raise ValueError("Site must lie on the selected GFS grid")
                values.append(full[row * ni + col])
            values = np.array(values)
            if not np.isfinite(values).all() or (np.abs(values) > 1e10).any():
                raise ValueError("Missing GFS values")
            return values
        finally:
            eccodes.codes_release(handle)


def download_gfs_object(job, resolution, sites, cache):
    cycle, hour = job
    key = gfs_key(cycle, hour, resolution)
    folder = Path(cache) / str(resolution) / f"{cycle:%Y%m%d%H}" / f"f{hour:03d}"
    inventory, index_metadata = fetch_cached(GFS_BASE + "/" + key + ".idx", folder / "inventory.idx")
    records = [line.split(":") for line in inventory.decode().splitlines()]
    values, metadata = [], []
    for name in FIELDS:
        selected = [i for i, row in enumerate(records) if row[3:5] == list(SPEC[name][:2])]
        if len(selected) != 1 or selected[0] == len(records) - 1:
            raise ValueError(f"Missing or ambiguous GFS inventory field: {name}")
        i = selected[0]
        span = (int(records[i][1]), int(records[i + 1][1]) - 1)
        payload, source = fetch_cached(GFS_BASE + "/" + key, folder / f"{name}.grib2", span)
        values.append(decode_field(payload, name, cycle, hour, resolution, sites))
        metadata.append(source)
    if len({item["etag"] for item in metadata}) != 1:
        raise ValueError("GFS fields belong to different versions of one object")
    available = max(pd.Timestamp(item["last_modified"]) for item in [index_metadata, *metadata])
    return {"values": np.array(values).T, "available_time": available, "key": key}


def source_jobs(issue, cycle, history_hours, resolution):
    times = pd.date_range(issue - pd.Timedelta(hours=history_hours - 1), periods=history_hours + 24, freq="h")
    jobs = []
    interval = 3 if resolution == 0.5 else 1
    for valid in times:
        source_cycle = cycle if valid >= cycle else cycle - pd.Timedelta(days=1)
        hour = int((valid - source_cycle) / pd.Timedelta(hours=1))
        lower, upper = hour // interval * interval, (hour + interval - 1) // interval * interval
        jobs.append(((source_cycle, lower), (source_cycle, upper), (hour - lower) / interval))
    return times, jobs


def download_gfs(
    issues, output, cache, resolution=0.5, sites=SITES, history_hours=0,
    workers=4, buffer_minutes=None, minimum_cycle_age=None, npz_output=None,
):
    import requests

    buffer_minutes = (15 if resolution == 0.25 else 0) if buffer_minutes is None else buffer_minutes
    minimum_cycle_age = (0 if resolution == 0.25 else 6) if minimum_cycle_age is None else minimum_cycle_age
    issues = pd.DatetimeIndex(pd.to_datetime(issues, utc=True))
    rows, windows, time_windows, available_windows, cycle_windows, fallbacks = [], [], [], [], [], []
    downloaded = {}
    for issue in issues:
        first_cycle = (issue - pd.Timedelta(hours=minimum_cycle_age)).floor("6h")
        for fallback in range(0, 49, 6):
            cycle = first_cycle - pd.Timedelta(hours=fallback)
            valid_times, jobs = source_jobs(issue, cycle, history_hours, resolution)
            needed = sorted({job for low, high, _ in jobs for job in (low, high)} - downloaded.keys())
            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for job, result in zip(needed, pool.map(
                        lambda job: download_gfs_object(job, resolution, sites, cache), needed
                    )):
                        downloaded[job] = result
            except requests.HTTPError as error:
                if error.response.status_code == 404:
                    continue
                raise
            available = [max(downloaded[low]["available_time"], downloaded[high]["available_time"]) for low, high, _ in jobs]
            if max(available) <= issue - pd.Timedelta(minutes=buffer_minutes):
                break
        else:
            raise ValueError(f"No complete GFS cycle available at {issue}")
        values = np.array([
            (1 - weight) * downloaded[low]["values"] + weight * downloaded[high]["values"]
            for low, high, weight in jobs
        ])
        for valid, (low, high, weight), availability, fields in zip(valid_times, jobs, available, values):
            for (site_id, latitude, longitude), point in zip(sites, fields):
                rows.append({
                    "issue_time": issue, "time": valid, "target_time": valid, "valid_time": valid,
                    "lead_hours": int((valid - issue) / pd.Timedelta(hours=1)),
                    "cycle_time": low[0], "forecast_hour": int((valid - low[0]) / pd.Timedelta(hours=1)),
                    "available_time": availability, "site_id": site_id,
                    "latitude": latitude, "longitude": longitude, **dict(zip(FIELDS, point)),
                    "lower_source_key": downloaded[low]["key"], "upper_source_key": downloaded[high]["key"],
                    "upper_weight": weight,
                })
        windows.append(values)
        time_windows.append(valid_times.tz_localize(None).to_numpy(dtype="datetime64[ns]"))
        available_windows.append(
            pd.DatetimeIndex(available).tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
        )
        cycle_windows.append(
            pd.DatetimeIndex([low[0] for low, _, _ in jobs]).tz_localize(None).to_numpy(dtype="datetime64[ns]")
        )
        fallbacks.append(fallback)
        print(f"GFS {issue.isoformat()}: cycle {cycle.isoformat()}, {len(valid_times)} hours", flush=True)
    frame = pd.DataFrame(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    write_json(output.with_suffix(".source.json"), {
        "source": GFS_BASE, "grid_degrees": resolution, "history_hours": history_hours,
        "forecast_hours": 24, "availability_buffer_minutes": buffer_minutes,
        "minimum_cycle_age_hours": minimum_cycle_age, "fallback_hours": fallbacks,
        "availability": "maximum S3 Last-Modified of GRIB and inventory objects",
        "interpolation": "raw meteorological fields, linear between 3-hour samples" if resolution == 0.5 else "native hourly",
    })
    if npz_output:
        Path(npz_output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_output, raw_values=np.array(windows), raw_columns=np.array(FIELDS),
            points_lat_lon=np.array([(lat, lon) for _, lat, lon in sites]),
            issue_times=issues.tz_localize(None).to_numpy(dtype="datetime64[ns]"), valid_times=np.array(time_windows),
            available_times=np.array(available_windows), cycle_times=np.array(cycle_windows),
            fallback_hours=np.array(fallbacks),
        )
    return frame
