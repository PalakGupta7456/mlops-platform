"""Production monitoring: forecast logging, live accuracy and data drift.

* Every hourly run logs the champion's 48-step forecast.
* When actuals arrive, logged forecasts are scored: live MAE vs ESO's forecast
  for exactly the same half-hours.
* Drift: Population Stability Index (PSI) of each input feature over the last
  7 days vs the model's training window, plus a KS test p-value.
* Alerts feed the retrain trigger: drift above the alert level or live MAE
  degrading past the holdout MAE forces a retrain in the next run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .config import DRIFT_PSI_ALERT, DRIFT_PSI_WARN, PRED_LOG

DRIFT_FEATURES = ["wind_mean", "solar_mean", "temp_mean", "cloud_mean", "actual"]


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference, current = reference[~np.isnan(reference)], current[~np.isnan(current)]
    if len(reference) < 20 or len(current) < 20:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / len(reference)
    c = np.histogram(current, edges)[0] / len(current)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(df: pd.DataFrame, train_end: pd.Timestamp, recent_days: int = 7) -> list[dict]:
    obs = df.dropna(subset=["actual"])
    ref = obs[obs["time"] <= train_end]
    cur = obs[obs["time"] > obs["time"].max() - pd.Timedelta(days=recent_days)]
    out = []
    for f in DRIFT_FEATURES:
        if f not in df:
            continue
        a, b = ref[f].to_numpy(float), cur[f].to_numpy(float)
        p = psi(a, b)
        status = "alert" if p >= DRIFT_PSI_ALERT else "warn" if p >= DRIFT_PSI_WARN else "ok"
        out.append({"feature": f, "psi": round(p, 4), "ks_p": round(float(ks_2samp(a, b).pvalue), 4) if len(a) and len(b) else None,
                    "ref_mean": round(float(np.nanmean(a)), 2), "cur_mean": round(float(np.nanmean(b)), 2), "status": status})
    return out


def log_forecast(fc: pd.DataFrame, path=PRED_LOG, keep_days: int = 45) -> pd.DataFrame:
    cols = ["issue_time", "target_time", "horizon", "pred", "eso_forecast", "model_version"]
    log = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=cols)
    log = pd.concat([log, fc[cols]], ignore_index=True)
    for c in ("issue_time", "target_time"):
        log[c] = pd.to_datetime(log[c])
    log = log.drop_duplicates(["issue_time", "horizon"], keep="last")
    log = log[pd.to_datetime(log["issue_time"]) >= pd.to_datetime(log["issue_time"]).max() - pd.Timedelta(days=keep_days)]
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_parquet(path, index=False)
    return log


def live_performance(log: pd.DataFrame, df: pd.DataFrame) -> dict:
    actual = df[["time", "actual"]].dropna().rename(columns={"time": "target_time"})
    log = log.assign(target_time=pd.to_datetime(log["target_time"]))
    s = log.merge(actual, on="target_time")
    if s.empty:
        return {"n_scored": 0}
    s = s.assign(err=(s["pred"] - s["actual"]).abs(), eso_err=(s["eso_forecast"].astype(float) - s["actual"]).abs(),
                 day=pd.to_datetime(s["target_time"]).dt.strftime("%Y-%m-%d"))
    daily = s.groupby("day")[["err", "eso_err"]].mean().round(2).tail(30)
    last7 = s[pd.to_datetime(s["target_time"]) > pd.to_datetime(s["target_time"]).max() - pd.Timedelta(days=7)]
    return {"n_scored": int(len(s)), "mae_7d": round(float(last7["err"].mean()), 2), "eso_mae_7d": round(float(last7["eso_err"].mean()), 2),
            "win_rate_7d": round(float((last7["err"] < last7["eso_err"]).mean()), 3),
            "daily": {"day": daily.index.tolist(), "model": daily["err"].tolist(), "eso": daily["eso_err"].tolist()}}


def retrain_reasons(drift: list[dict], live: dict, holdout_mae: float | None, days_since_train: float) -> list[str]:
    reasons = []
    if days_since_train >= 1:
        reasons.append("schedule: daily retrain")
    alerts = [d["feature"] for d in drift if d["status"] == "alert"]
    if alerts:
        reasons.append(f"drift: PSI ≥ {DRIFT_PSI_ALERT} on {', '.join(alerts)}")
    if holdout_mae and live.get("n_scored", 0) > 200 and live["mae_7d"] > 1.25 * holdout_mae:
        reasons.append(f"performance: live MAE {live['mae_7d']} > 1.25 × holdout {holdout_mae}")
    return reasons
