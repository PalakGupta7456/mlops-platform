"""Direct multi-horizon features: what is known at issue time t, plus the
weather forecast for the target half-hour t+h."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HORIZONS, LAGS

WEATHER = ["wind_mean", "wind_cubed", "solar_mean", "temp_mean", "cloud_mean"]


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
        base[f"ci_lag_{lag}"] = y.shift(lag)
    for w in (4, 12, 48):
        base[f"ci_mean_{w}"] = y.rolling(w, min_periods=1).mean()
    base["ci_std_48"] = y.rolling(48, min_periods=4).std()
    base["ci_diff_1"] = y.diff()
    base["wind_now"] = df["wind_mean"]
    tgt = df[["time", "actual", "eso_forecast", *WEATHER]].rename(columns={"time": "target_time", "actual": "target"})
    tgt = tgt.rename(columns={c: f"tgt_{c}" for c in WEATHER})
    blocks = []
    for h in horizons:
        b = base.copy()
        b["horizon"] = h
        b["target_time"] = b["time"] + pd.Timedelta(minutes=30 * h)
        b = b.merge(tgt, on="target_time", how="left")
        b["tgt_wind_change"] = b["tgt_wind_mean"] - b["wind_now"]
        blocks.append(pd.concat([b.reset_index(drop=True), _calendar(b["target_time"], "tgt_")], axis=1))
    return pd.concat(blocks, ignore_index=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in {"time", "target_time", "target", "eso_forecast"}]
