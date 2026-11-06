"""Live data: GB carbon intensity (National Grid ESO API) + weather (Open-Meteo).

The ESO API returns, for every half hour, the *actual* intensity once known
and ESO's own *forecast*: our model's natural benchmark.
"""
from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import requests

from .config import CARBON_API, DATA_FILE, HISTORY_DAYS, WEATHER_API, WEATHER_POINTS, WEATHER_VARS

log = logging.getLogger(__name__)


def _get(url: str, params: dict | None = None, retries: int = 4) -> dict | list:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60, headers={"Accept": "application/json"})
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:  # pragma: no cover - network
            log.warning("%s (attempt %d)", exc, attempt + 1)
            time.sleep(2 * 2**attempt)
    raise RuntimeError(f"failed: {url}")


def fetch_intensity(days: int = HISTORY_DAYS) -> pd.DataFrame:
    """Half-hourly actual + ESO forecast, past `days` plus the next 48 h (actual = NaN)."""
    end = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) + timedelta(hours=48)
    cur = end - timedelta(days=days + 2)
    rows = []
    while cur < end:
        nxt = min(cur + timedelta(days=13), end)
        payload = _get(f"{CARBON_API}/intensity/{cur:%Y-%m-%dT%H:%MZ}/{nxt:%Y-%m-%dT%H:%MZ}")
        for d in payload["data"]:
            rows.append({"time": d["from"], "actual": d["intensity"].get("actual"), "eso_forecast": d["intensity"].get("forecast")})
        cur = nxt
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None)
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def fetch_weather(days: int = HISTORY_DAYS) -> pd.DataFrame:
    """Hourly weather at each point, interpolated to half hours, averaged into national features."""
    lats = ",".join(str(p[0]) for p in WEATHER_POINTS.values())
    lons = ",".join(str(p[1]) for p in WEATHER_POINTS.values())
    payload = _get(WEATHER_API, {"latitude": lats, "longitude": lons, "hourly": ",".join(WEATHER_VARS),
                                 "past_days": min(days + 2, 92), "forecast_days": 3, "timezone": "UTC"})
    payload = payload if isinstance(payload, list) else [payload]
    frames = []
    for name, p in zip(WEATHER_POINTS, payload, strict=True):
        h = pd.DataFrame(p["hourly"])
        h["time"] = pd.to_datetime(h["time"])
        h = h.set_index("time").resample("30min").interpolate()
        frames.append(h.add_prefix(f"{name}_"))
    w = pd.concat(frames, axis=1)
    w["wind_mean"] = w[[c for c in w if c.endswith("wind_speed_100m")]].mean(axis=1)
    w["wind_cubed"] = (w[[c for c in w if c.endswith("wind_speed_100m")]] ** 3).mean(axis=1) / 1000  # power ∝ v³
    w["solar_mean"] = w[[c for c in w if c.endswith("shortwave_radiation")]].mean(axis=1)
    w["temp_mean"] = w[[c for c in w if c.endswith("temperature_2m")]].mean(axis=1)
    w["cloud_mean"] = w[[c for c in w if c.endswith("cloud_cover")]].mean(axis=1)
    return w.reset_index()


def fetch() -> pd.DataFrame:
    df = fetch_intensity().merge(fetch_weather(), on="time", how="left")
    return df


def update_store(new: pd.DataFrame, path=DATA_FILE, keep_days: int = HISTORY_DAYS + 30) -> pd.DataFrame:
    """Upsert by timestamp: newer pulls overwrite (actuals arrive after forecasts)."""
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old[~old["time"].isin(new["time"])], new], ignore_index=True)
    new = new.sort_values("time")
    new = new[new["time"] >= new["time"].max() - pd.Timedelta(days=keep_days)].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False)
    return new


def synthetic(days: int = 60, seed: int = 0, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Realistic stand-in: intensity falls with wind power and sun, rises with cold and evening demand."""
    rng = np.random.default_rng(seed)
    end = end or pd.Timestamp.now().floor("30min")
    t = pd.date_range(end - pd.Timedelta(days=days), end + pd.Timedelta(hours=48), freq="30min")
    n = len(t)
    hod = (t.hour + t.minute / 60).to_numpy()
    wind = np.empty(n)
    w = 9.0
    for i in range(n):
        w = max(0.5, w + 0.03 * (9 - w) + rng.normal(0, 0.45))
        wind[i] = w
    solar = np.clip(np.sin(np.pi * (hod - 7) / 10), 0, None) * rng.uniform(150, 450, n)
    temp = 8 + 4 * np.sin(np.pi * (hod - 9) / 12) + rng.normal(0, 1, n)
    demand = 1 + 0.25 * np.exp(-((hod - 18) ** 2) / 4) - 0.15 * (t.dayofweek.to_numpy() >= 5) + 0.01 * (12 - temp)
    actual = np.clip(260 * demand - 6.5 * np.minimum(wind, 18) - 0.12 * solar + rng.normal(0, 10, n), 15, None)
    eso = actual + rng.normal(0, 22, n)
    df = pd.DataFrame({"time": t, "actual": actual.round(), "eso_forecast": eso.round(),
                       "wind_mean": wind, "wind_cubed": wind**3 / 1000, "solar_mean": solar, "temp_mean": temp,
                       "cloud_mean": rng.uniform(20, 100, n)})
    df.loc[df["time"] > end, "actual"] = np.nan
    return df
