"""Residual LSTM ensembles with their original checkpoint normalization."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCALE_KW = 5000.0


class ConditionalLSTM(nn.Module):
    def __init__(self, hidden_size=8, residual=True):
        super().__init__()
        self.residual = residual
        self.encoder = nn.LSTM(4, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)
        if residual:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, weather, curve):
        hidden, _ = self.encoder(weather)
        output = self.head(hidden).squeeze(-1)
        return output + curve if self.residual else output


def predict(model_dir, x, kind):
    """Return ensemble-mean kW for [windows, 24 hours, 4 physical features]."""
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[1:] != (24, 4) or not np.isfinite(x).all():
        raise ValueError("Expected finite features with shape [windows, 24, 4].")
    if kind not in ("era5", "gfs"):
        raise ValueError("Model kind must be 'era5' or 'gfs'.")
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "model.json").read_text())
    checkpoints = [model_dir / name for name in manifest["checkpoints"]]
    members = []
    for path in checkpoints:
        record = torch.load(path, map_location="cpu", weights_only=True)
        if kind == "era5":
            config = record["model_config"]
            model = ConditionalLSTM(config["hidden_size"], config["residual"])
            model.load_state_dict(record["model_state_dict"])
            values = x.astype(np.float64).copy()
            values[:, :, 3] /= SCALE_KW
            normalized = (values - np.array(record["scaler_mean"])) / np.array(record["scaler_std"])
            curve = values[:, :, 3].astype(np.float32)
        else:
            model = ConditionalLSTM(record["config"]["hidden"])
            model.load_state_dict(record["state"])
            scaler = record["scaler"]
            normalized = (x.astype(np.float64) - np.array(scaler["mean"])) / np.array(scaler["scale"])
            curve = np.asarray(x[:, :, 3] / SCALE_KW, dtype=np.float32)
        model.eval()
        with torch.inference_mode():
            output = model(torch.from_numpy(normalized.astype(np.float32)), torch.from_numpy(curve))
            members.append(output.numpy() * SCALE_KW)
    return np.mean(members, axis=0)
