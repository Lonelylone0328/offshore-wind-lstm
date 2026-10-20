"""Train the selected annual weather and power correction architectures."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .correction import (
    STATE_SCALES,
    FrozenPower,
    PowerResidualLSTM,
    WeatherLSTM,
    inputs,
    load_dataset,
)


def split_days(data):
    days = pd.DatetimeIndex(data["valid_times"][:, 24])
    if len(set(days.year)) != 1 or not days.equals(
            pd.date_range(f"{days[0].year}-01-01", f"{days[0].year}-12-31", freq="D")):
        raise ValueError("Annual training requires one complete calendar year.")
    remaining = days.days_in_month - days.day
    return np.array(remaining >= 6), np.array(remaining < 5)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) ** 2)))


def train(route, data_path, target_path, output, teacher_dir, max_epochs=None):
    if route not in {"weather", "power"}:
        raise ValueError("Choose weather or power.")
    torch.set_num_threads(1)
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    data = load_dataset(data_path)
    train_mask, valid = split_days(data)
    raw, base_state, columns = inputs(data)
    with np.load(target_path, allow_pickle=False) as archive:
        truth = archive["state"].copy()
        if not np.array_equal(archive["issue_times"], data["issue_times"]):
            raise ValueError("Weather targets and training origins do not align.")
    mean, std = raw[train_mask].mean((0, 1)), raw[train_mask].std((0, 1))
    std[std < 1e-8] = 1.
    teacher = FrozenPower(teacher_dir)
    tx = torch.tensor((raw - mean) / std, dtype=torch.float32)
    state_tensor, truth_tensor = [torch.tensor(a, dtype=torch.float32) for a in (base_state, truth)]
    with torch.no_grad():
        power_base, power_target = teacher(state_tensor), teacher(truth_tensor)
    scale = torch.tensor(STATE_SCALES, dtype=torch.float32)
    indices = np.flatnonzero(train_mask)
    epochs = 500 if max_epochs is None else max_epochs
    config = {"hidden_size": 16, "bidirectional": True}
    if route == "weather":
        config["power_weight"] = 4.
    members, predictions = [], []
    for seed in (20260910, 20260911, 20260912):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        cls = WeatherLSTM if route == "weather" else PowerResidualLSTM
        model = cls(raw.shape[-1], config["hidden_size"], config["bidirectional"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.003)
        best = float("inf") if route == "weather" else _rmse(
            power_base[valid].numpy().clip(0., 5000.), power_target[valid].numpy())
        best_epoch, stale = 0, 0
        best_state = copy.deepcopy(model.state_dict())
        history = [] if route == "weather" else [{"epoch": 0, "validation_rmse_kw": best}]
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for batch in np.array_split(rng.permutation(indices), int(np.ceil(len(indices) / 32))):
                if route == "weather":
                    prediction = model(tx[batch], state_tensor[batch])
                    loss = ((prediction - truth_tensor[batch]) / scale).square().mean()
                    loss += config["power_weight"] * (
                        teacher(prediction) / 5000. - power_target[batch] / 5000.).square().mean()
                else:
                    prediction = model(tx[batch], power_base[batch], bounded=False)
                    loss = ((prediction - power_target[batch]) / 5000.).square().mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            with torch.no_grad():
                if route == "weather":
                    val_power = teacher(model(tx[valid], state_tensor[valid])) / 5000.
                    value = float((val_power - power_target[valid] / 5000.).square().mean().sqrt() * 5000.)
                else:
                    value = _rmse(model(tx[valid], power_base[valid]).numpy(), power_target[valid].numpy())
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_rmse_kw": value})
            if value < best - 1e-5:
                best, best_epoch, best_state, stale = value, epoch, copy.deepcopy(model.state_dict()), 0
            else:
                stale += 1
            if stale >= 65:
                break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            base = state_tensor if route == "weather" else power_base
            predictions.append(model(tx, base).numpy())
        name = f"{route}_seed{seed}.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": config, "seed": seed,
                    "best_epoch": best_epoch, "scaler_mean": mean.tolist(),
                    "scaler_std": std.tolist(), "columns": columns}, folder / name)
        (folder / f"{route}_seed{seed}_history.json").write_text(json.dumps(history, indent=2) + "\n")
        members.append({"checkpoint": name, "seed": seed, "best_epoch": best_epoch,
                        "validation_rmse_kw": best})
    with torch.no_grad():
        if route == "weather":
            ensemble = teacher(torch.tensor(np.mean(predictions, axis=0), dtype=torch.float32)).numpy()
        else:
            ensemble = np.mean(predictions, axis=0, dtype=np.float64)
    record = {"route": route, "checkpoints": [row["checkpoint"] for row in members],
              "config": config, "members": members,
              "train_origins": int(train_mask.sum()), "validation_origins": int(valid.sum()),
              "validation_rmse_to_teacher_kw": _rmse(ensemble[valid], power_target[valid].numpy())}
    (folder / "model.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
