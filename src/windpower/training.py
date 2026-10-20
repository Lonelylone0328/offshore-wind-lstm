"""Train the selected ERA5 conditional and GFS forecast model configurations."""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .models import SCALE_KW, ConditionalLSTM
from .physics import NAMES


def train(kind, data_path, output, max_epochs=None):
    """Train 3 ERA5 members with validation, or 5 GFS members for 23 epochs."""
    if kind not in ("era5", "gfs"):
        raise ValueError("Model kind must be 'era5' or 'gfs'.")
    with np.load(data_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    x, y = data["x"], np.asarray(data["y"], dtype=float)
    if x.ndim != 3 or x.shape[1:] != (24, 4) or y.shape != x.shape[:2]:
        raise ValueError("Expected x [windows, 24, 4] and y [windows, 24].")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Training data must be finite.")
    training = np.flatnonzero(data["train_mask"])
    validation = np.flatnonzero(data["validation_mask"])
    if not len(training) or np.intersect1d(training, validation).size:
        raise ValueError("Training must be nonempty and separate from validation.")
    if kind == "era5" and not len(validation):
        raise ValueError("ERA5 early stopping requires validation windows.")
    if len(validation) and np.intersect1d(
        data["target_times"][training], data["target_times"][validation]
    ).size:
        raise ValueError("Training and validation target hours overlap.")
    epochs = max_epochs if max_epochs is not None else (600 if kind == "era5" else 23)
    if epochs < 1:
        raise ValueError("Epoch count must be positive.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    values = x.astype(np.float64).copy()
    if kind == "era5":
        values[:, :, 3] /= SCALE_KW
    fit_values = values[training].reshape(-1, 4)
    if kind == "era5":
        _, first = np.unique(data["target_times"][training].ravel(), return_index=True)
        fit_values = fit_values[np.sort(first)]
    mean, std = fit_values.mean(axis=0), fit_values.std(axis=0)
    std[std < 1e-12] = 1.0
    xt = torch.from_numpy(((values - mean) / std).astype(np.float32))
    curve = values[:, :, 3] if kind == "era5" else x[:, :, 3] / SCALE_KW
    ct = torch.from_numpy(curve.astype(np.float32))
    yt = torch.from_numpy((y / SCALE_KW).astype(np.float32))
    seeds = range(20260907, 20260910) if kind == "era5" else range(20260991, 20260996)
    paths = []
    for seed in seeds:
        torch.manual_seed(seed)
        numpy_rng = np.random.default_rng(seed)
        torch_rng = torch.Generator().manual_seed(seed)
        model = ConditionalLSTM()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.003 if kind == "era5" else 0.001,
            weight_decay=0.0001 if kind == "era5" else 0.001,
        )
        history, best, best_epoch, steps = [], float("inf"), 0, 0
        for epoch in range(1, epochs + 1):
            model.train()
            if kind == "era5":
                order = numpy_rng.permutation(training)
                batches = [order[start:start + 32] for start in range(0, len(order), 32)]
            else:
                order = torch.as_tensor(training)[torch.randperm(len(training), generator=torch_rng)]
                batches = order.split(4)
            for indices in batches:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(xt[indices], ct[indices]), yt[indices])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                steps += 1
            model.eval()
            with torch.no_grad():
                train_loss = nn.functional.mse_loss(model(xt[training], ct[training]), yt[training]).item()
                validation_loss = (
                    nn.functional.mse_loss(model(xt[validation], ct[validation]), yt[validation]).item()
                    if kind == "era5" else None
                )
            history.append({
                "epoch": epoch, "train_rmse_kw": train_loss**0.5 * SCALE_KW,
                "validation_rmse_kw": validation_loss**0.5 * SCALE_KW if validation_loss is not None else None,
            })
            if kind == "gfs" or validation_loss < best - 1e-12:
                best = validation_loss if kind == "era5" else train_loss
                best_epoch = epoch
                state = copy.deepcopy(model.state_dict())
            if kind == "era5" and epoch - best_epoch >= 80:
                break
        record = {
            "seed": seed, "training_steps": steps, "best_epoch": best_epoch, "history": history,
        }
        if kind == "era5":
            record.update({
                "model_config": {"hidden_size": 8, "residual": True},
                "model_state_dict": state,
                "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
                "target_scale_kw": SCALE_KW,
            })
        else:
            record.update({
                "trained": True, "config": {"feature_set": "physics", "hidden": 8, "lr": 0.001},
                "state": state, "scaler": {"mean": mean.tolist(), "scale": std.tolist(), "fit_split": "train"},
                "features": NAMES, "scale_kw": SCALE_KW, "horizon": 24,
                "fixed_epochs": epochs,
            })
        path = output / f"{kind}_seed_{seed}.pt"
        torch.save(record, path)
        paths.append(path)
    (output / "model.json").write_text(json.dumps({
        "kind": kind, "checkpoints": [path.name for path in paths], "features": NAMES,
    }, indent=2) + "\n")
    return paths
