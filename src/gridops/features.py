"""Direct multi-horizon features, made scale-free on purpose.

Lesson learned on live data: with absolute features (current intensity, raw
lags, raw weather), a model trained on August-October learned that 230+
gCO2/kWh is a rare spike that always falls. In November that level is normal,
so the model forecast a crash every hour and lost to "no change". All history
and weather features are therefore expressed *relative to now*, and the target
is the change from now. The model learns dynamics, not levels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZONS, LAGS

WEATHER = ["wind_mean", "wind_cubed", "solar_mean", "temp_mean", "cloud_mean"]
CALENDAR = ["tgt_hod_sin", "tgt_hod_cos", "tgt_dow_sin", "tgt_dow_cos", "tgt_weekend"]


def _calendar(t: pd.Series, prefix: str) -> pd.DataFrame:
    hod = t.dt.hour + t.dt.minute / 60
    dow = t.dt.dayofweek
    return pd.DataFrame({f"{prefix}hod_sin": np.sin(2 * np.pi * hod / 24), f"{prefix}hod_cos": np.cos(2 * np.pi * hod / 24),
                         f"{prefix}dow_sin": np.sin(2 * np.pi * dow / 7), f"{prefix}dow_cos": np.cos(2 * np.pi * dow / 7),
                         f"{prefix}weekend": (dow >= 5).astype(int)})


def build(df: pd.DataFrame, horizons: list[int] = HORIZONS) -> pd.DataFrame:
    df = df.sort_values("time").reset_index(drop=True)
    y = df["actual"].astype(float)
    base = pd.DataFrame({"time": df["time"], "ci_now": y})
    for lag in LAGS:
        base[f"ci_lag_{lag}"] = y - y.shift(lag)                 # change since `lag` steps ago
    for w in (4, 12, 48):
        base[f"ci_mean_{w}"] = y - y.rolling(w, min_periods=1).mean()
    base["ci_std_48"] = y.rolling(48, min_periods=4).std()
    base["ci_diff_1"] = y.diff()
    for w in WEATHER:
        base[f"{w}_now"] = df[w].to_numpy()
    tgt = df[["time", "actual", "eso_forecast", *WEATHER]].rename(columns={"time": "target_time", "actual": "target"})
    tgt = tgt.rename(columns={c: f"tgt_{c}" for c in WEATHER})
    blocks = []
    for h in horizons:
        b = base.copy()
        b["horizon"] = h
        b["target_time"] = b["time"] + pd.Timedelta(minutes=30 * h)
        b = b.merge(tgt, on="target_time", how="left")
        for w in WEATHER:
            b[f"d_{w}"] = b[f"tgt_{w}"] - b[f"{w}_now"]          # forecast weather change between now and target
        blocks.append(pd.concat([b.reset_index(drop=True), _calendar(b["target_time"], "tgt_")], axis=1))
    return pd.concat(blocks, ignore_index=True)


def feature_columns(frame: pd.DataFrame | None = None) -> list[str]:
    lags = [f"ci_lag_{lag}" for lag in LAGS] + [f"ci_mean_{w}" for w in (4, 12, 48)]
    return lags + ["ci_std_48", "ci_diff_1", "horizon"] + [f"d_{w}" for w in WEATHER] + CALENDAR


class PersistenceBlend:
    """prediction = now + w[h] * model_delta.

    w[h] is fitted per horizon on a validation window: least squares of the true
    change on the model's predicted change, clipped to [0, 1]. Where the model has
    no edge (very short horizons), w goes to 0 and the forecast falls back to
    persistence; where it helps, it is trusted more.
    """

    def __init__(self, model, weights: dict[int, float], features: list[str]):
        self.model, self.weights, self.features = model, {int(k): float(v) for k, v in weights.items()}, features

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        delta = self.model.predict(X[self.features])
        w = X["horizon"].map(self.weights).fillna(0.5).to_numpy()
        return X["ci_now"].to_numpy(dtype=float) + w * delta

    @property
    def input_columns(self) -> list[str]:
        return ["ci_now", *self.features]

    @staticmethod
    def fit_weights(delta_pred: np.ndarray, frame: pd.DataFrame) -> dict[int, float]:
        out = {}
        d = frame.assign(_dm=delta_pred, _dy=frame["target"] - frame["ci_now"])
        for h, g in d.groupby("horizon"):
            x, yv = g["_dm"].to_numpy(), g["_dy"].to_numpy()
            out[int(h)] = float(np.clip((x @ yv) / max(float(x @ x), 1e-9), 0, 1))
        return out
