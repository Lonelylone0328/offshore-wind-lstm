"""NREL 5 MW OC3 monopile labels from OpenFAST and TurbSim."""

import json
import os
import re
import shutil
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

OPENFAST_COMMIT = "2895884d2be01862173c88d70f86b358d2f1a50a"
RTEST_COMMIT = "dd5feaaaa500ba7283140107806300d551cff0a7"
REFERENCE_CASE = "5MW_OC3Mnpl_DLL_WTurb_WavesIrr"
CONTROLLER_SOURCE = "glue-codes/openfast/5MW_Baseline/ServoData/DISCON/DISCON.F90"


@dataclass(frozen=True)
class SimulationCase:
    case_id: str
    time: str
    hub_wind_speed_mps: float
    air_density_kgm3: float = 1.225
    shear_exponent: float = 0.14
    turbulence_intensity: float = 0.08
    seed: int = 20260906
    duration_s: float = 720.0
    warmup_s: float = 120.0

    @property
    def parked(self):
        return self.hub_wind_speed_mps < 3 or self.hub_wind_speed_mps >= 25


@dataclass(frozen=True)
class Installation:
    openfast: Path
    turbsim: Path
    controller: Path
    templates: Path
    turbsim_template: Path

    @classmethod
    def from_prefix(cls, prefix):
        prefix = Path(prefix).resolve()
        return cls(
            prefix / "build/glue-codes/openfast/openfast",
            prefix / "build/modules/turbsim/turbsim",
            prefix / "DISCON.dll",
            prefix / "r-test/glue-codes/openfast",
            prefix / "openfast/docs/source/user/turbsim/examples/TurbSim.inp",
        )


def replace_parameter(path, name, value):
    text = path.read_text()
    pattern = re.compile(r'^(\s*)(?:"[^"]*"|\S+)(\s+' + re.escape(name) + r"\s+.*)$", re.MULTILINE)
    text, count = pattern.subn(lambda match: f"{match[1]}{value}{match[2]}", text)
    if count != 1:
        raise ValueError(f"{path}: expected one {name} parameter, found {count}")
    path.write_text(text)


def quoted(path):
    return '"' + str(path).replace("\\", "/") + '"'


def prepare_case(case, installation, output_dir):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", case.case_id):
        raise ValueError("case_id must contain letters, numbers, underscores or hyphens")
    if case.duration_s - case.warmup_s < 600 or case.warmup_s < 0:
        raise ValueError("At least 600 seconds after warmup are required")
    if not -0.2 <= case.shear_exponent <= 0.5:
        raise ValueError("Use the physical feature preprocessing for shear in [-0.2, 0.5]")
    outdir = Path(output_dir).resolve() / case.case_id
    model = outdir / "model"
    model.mkdir(parents=True, exist_ok=True)
    for name in ("5MW_Baseline", "Wind"):
        link = outdir / name
        if not link.exists():
            link.symlink_to(installation.templates / name, target_is_directory=True)
    for source in (installation.templates / REFERENCE_CASE).iterdir():
        if source.suffix in (".dat", ".fst"):
            shutil.copy2(source, model / source.name)
    fst = model / f"{REFERENCE_CASE}.fst"
    servo = model / "NRELOffshrBsline5MW_OC3Monopile_ServoDyn.dat"
    elast = model / "NRELOffshrBsline5MW_OC3Monopile_ElastoDyn.dat"
    sea = model / "SeaState.dat"
    inflow = model / "InflowWind.dat"
    shutil.copy2(installation.templates / "5MW_Baseline/NRELOffshrBsline5MW_InflowWind_12mps.dat", inflow)
    replace_parameter(servo, "DLL_FileName", quoted(installation.controller))
    for key, value in {
        "TMax": case.duration_s, "DT": 0.01, "ModCoupling": 3, "DT_Out": 0.1,
        "OutFileFmt": 1, "SttsTime": 60, "AirDens": case.air_density_kgm3,
        "InflowFile": quoted(inflow),
    }.items():
        replace_parameter(fst, key, value)
    for key, value in {
        "WaveMod": 0, "WaveHs": 0, "WaveTp": 8, "WaveTMax": case.duration_s,
        "WaveSeed(1)": case.seed, "WvDiffQTF": "False", "WvSumQTF": "False",
    }.items():
        replace_parameter(sea, key, value)
    for key, value in {
        "PropagationDir": 0, "RefHt": 90, "PLExp": case.shear_exponent,
        "HWindSpeed": case.hub_wind_speed_mps,
    }.items():
        replace_parameter(inflow, key, value)
    if case.parked:
        for key, value in {"RotSpeed": 0, "GenDOF": "False", "DrTrDOF": "False"}.items():
            replace_parameter(elast, key, value)
        for key, value in {
            "PCMode": 0, "VSContrl": 1, "TimGenOn": 999999, "VS_RtGnSp": 1173.7,
            "VS_Rgn2K": 0.0, "VS_RtTq": 43093.55, "VS_SlPc": 10.0,
        }.items():
            replace_parameter(servo, key, value)
        for blade in range(1, 4):
            replace_parameter(elast, f"BlPitch({blade})", 90)
            replace_parameter(servo, f"PitNeut({blade})", 90)
    else:
        replace_parameter(elast, "RotSpeed", min(12.1, max(6.9, case.hub_wind_speed_mps * 1.02)))
        pitch = max(0.0, min(23.0, (case.hub_wind_speed_mps - 11.4) * 1.8))
        for blade in range(1, 4):
            replace_parameter(elast, f"BlPitch({blade})", pitch)
    if case.hub_wind_speed_mps >= 0.5:
        ts = model / "TurbSim.inp"
        shutil.copy2(installation.turbsim_template, ts)
        # Equal 170 m sides give a 5 m base in this TurbSim version.
        for key, value in {
            "RandSeed1": case.seed, "NumGrid_Z": 15, "NumGrid_Y": 15,
            "TimeStep": 0.1, "AnalysisTime": case.duration_s, "UsableTime": case.duration_s,
            "HubHt": 90, "GridHeight": 170, "GridWidth": 170, "TurbModel": '"IECKAI"',
            "ScaleIEC": 1, "IECstandard": '"1-Ed3"',
            "IECturbc": quoted(f"{100 * case.turbulence_intensity:.5f}"),
            "WindProfileType": '"PL"', "RefHt": 90, "URef": case.hub_wind_speed_mps,
            "PLExp": case.shear_exponent,
        }.items():
            replace_parameter(ts, key, value)
        replace_parameter(inflow, "WindType", 3)
        replace_parameter(inflow, "FileName_BTS", quoted(model / "TurbSim.bts"))
    else:
        replace_parameter(inflow, "WindType", 1)
    (outdir / "case.json").write_text(json.dumps(asdict(case), indent=2) + "\n")
    return fst


def inspect_wind_field(path, case):
    with Path(path).open("rb") as stream:
        header = struct.unpack("<h4i12fi", stream.read(70))
    kind, nz, ny, ntower, nt = header[:5]
    dz, dy, dt, wind, hub, base = header[5:11]
    expected_size = 70 + header[-1] + nt * 3 * (ny * nz + ntower) * 2
    if Path(path).stat().st_size != expected_size:
        raise ValueError("Incomplete TurbSim field")
    if (kind, nz, ny, ntower) != (7, 15, 15, 0) or not np.allclose(
        [dz * 14, dy * 14, dt, wind, hub, base],
        [170, 170, 0.1, case.hub_wind_speed_mps, 90, 5], rtol=1e-6, atol=1e-5,
    ):
        raise ValueError("TurbSim geometry does not match the 170 m grid")
    # Nonperiodic InflowWind shifts time by (half-width - x) / U.
    required_end = case.duration_s + (85 + 63 + 5) / wind
    if (nt - 1) * dt < required_end - 1e-4:
        raise ValueError("TurbSim field ends before the Taylor-shifted simulation")
    return {"base_height_m": base, "top_height_m": base + 14 * dz, "last_time_s": (nt - 1) * dt}


def summarize_output(path, case):
    lines = Path(path).read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.split()[:1] == ["Time"])
    names = lines[header].split()
    units = dict(zip(names, lines[header + 1].split()))
    values = np.loadtxt(lines[header + 2:])
    frame = pd.DataFrame(values, columns=names)
    times = frame.Time.to_numpy()
    if units["GenPwr"] != "(kW)" or not np.isfinite(values).all():
        raise ValueError("Invalid OpenFAST power units or nonfinite output")
    if not np.isclose(times[0], 0) or not np.isclose(times[-1], case.duration_s) or not np.allclose(
        np.diff(times), 0.1, rtol=0, atol=1e-5
    ):
        raise ValueError("OpenFAST output must cover the full simulation at 0.1 s intervals")
    retained = frame.loc[frame.Time > case.warmup_s]
    if len(retained) != round((case.duration_s - case.warmup_s) / 0.1):
        raise ValueError("Incomplete post-warmup output")
    pitch = frame.BldPitch1.to_numpy()
    if pitch.min() < -0.1 or pitch.max() > 90.1 or np.abs(np.diff(pitch)).max() / 0.1 > 8.1:
        raise ValueError("Controller pitch angle or rate exceeded its operating limits")
    if case.parked:
        if not (
            np.allclose(retained.GenPwr, 0) and np.allclose(retained.RotSpeed, 0)
            and np.allclose(retained.BldPitch1, 90, atol=0.01)
        ):
            raise ValueError("Parked case must have 0 kW, 0 rpm and 90-degree pitch")
    elif frame.RotSpeed.min() < -0.1 or frame.RotSpeed.max() > 20:
        raise ValueError("Rotor speed exceeded its operating range")
    power = retained.GenPwr.to_numpy()
    midpoint = len(power) // 2
    return {
        "target_power_kw": float(power.mean()), "power_std_kw": float(power.std()),
        "retained_samples": len(power), "effective_seconds": len(power) * 0.1,
        "half_mean_difference_kw": float(power[midpoint:].mean() - power[:midpoint].mean()),
    }


def run_solver(command, model, log):
    with log.open("w") as stream:
        subprocess.run(
            list(map(str, command)), cwd=model, stdout=stream, stderr=subprocess.STDOUT,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            check=True, timeout=1800,
        )


def run_case(case, installation, output_dir):
    outdir = Path(output_dir).resolve() / case.case_id
    result_path = outdir / "result.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved["case"] != asdict(case):
            raise ValueError(f"Existing case has different inputs: {case.case_id}")
        return saved
    fst = prepare_case(case, installation, output_dir)
    model = fst.parent
    geometry = None
    if case.hub_wind_speed_mps >= 0.5:
        run_solver([installation.turbsim, model / "TurbSim.inp"], model, outdir / "turbsim.log")
        geometry = inspect_wind_field(model / "TurbSim.bts", case)
    run_solver([installation.openfast, fst], model, outdir / "openfast.log")
    log = (outdir / "openfast.log").read_text()
    errors = ("Failed to converge in", "Small angle assumption violated", "Mach number exceeds 0.3",
              "Tower strike", "FATAL ERROR", "encountered an error")
    if "OpenFAST terminated normally." not in log or any(error.lower() in log.lower() for error in errors):
        raise ValueError(f"OpenFAST did not complete a valid run: {outdir / 'openfast.log'}")
    result = {
        "case": asdict(case), **summarize_output(fst.with_suffix(".out"), case),
        "wind_geometry": geometry, "openfast_commit": OPENFAST_COMMIT, "rtest_commit": RTEST_COMMIT,
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    return result
