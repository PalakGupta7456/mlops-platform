"""Hourly control loop: ingest -> (retrain if triggered) -> forecast -> monitor -> publish."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from . import data, monitoring, registry
from .config import DATA_FILE, PRED_LOG, SITE_DATA_DIR, STATE_DIR
from .features import build

log = logging.getLogger("gridops")
RUNS_FILE = STATE_DIR / "runs.json"


def forecast(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    obs = df.dropna(subset=["actual"])
    issue = obs["time"].max()
    frame = build(df)
    frame = frame[frame["time"] == issue]
    pred = np.clip(bundle["model"].predict(frame[bundle["features"]]), 0, None)
    return pd.DataFrame({"issue_time": frame["time"].values, "target_time": frame["target_time"].values, "horizon": frame["horizon"].values,
                         "pred": pred.round(1), "eso_forecast": frame["eso_forecast"].values, "model_version": bundle["version"]})


def run(df_new: pd.DataFrame | None = None, force_retrain: bool = False) -> dict:
    t0 = datetime.now(UTC)
    df = data.update_store(df_new if df_new is not None else data.fetch(), DATA_FILE)
    serving = STATE_DIR / "serving" / "model.json"
    meta = json.loads(serving.read_text()) if serving.exists() else None

    train_end = df.dropna(subset=["actual"])["time"].max() - pd.Timedelta(days=7)
    drift = monitoring.drift_report(df, train_end)
    plog = pd.read_parquet(PRED_LOG) if PRED_LOG.exists() else pd.DataFrame()
    live = monitoring.live_performance(plog, df) if len(plog) else {"n_scored": 0}
    days_since = (t0 - datetime.fromisoformat(meta["exported_at"])).total_seconds() / 86400 if meta else 99
    reasons = ["bootstrap: no champion yet"] if meta is None else monitoring.retrain_reasons(drift, live, meta["metrics"]["mae"], days_since)
    if force_retrain:
        reasons.append("manual trigger")

    decision = None
    if reasons:
        decision = registry.train_and_register(df, reason="; ".join(reasons))
        decision["reasons"] = reasons
        meta = json.loads(serving.read_text())

    bundle = registry.load_serving()
    fc = forecast(df, bundle)
    plog = monitoring.log_forecast(fc, PRED_LOG)
    live = monitoring.live_performance(plog, df)

    runs = json.loads(RUNS_FILE.read_text()) if RUNS_FILE.exists() else []
    runs.append({"at": t0.isoformat(timespec="minutes"), "seconds": round((datetime.now(UTC) - t0).total_seconds(), 1),
                 "retrained": decision is not None, "promoted": bool(decision and decision["promoted"]),
                 "reasons": reasons, "champion": bundle["version"]})
    runs = runs[-72:]
    RUNS_FILE.write_text(json.dumps(runs))
    payload = publish(df, fc, meta, live, drift, runs, decision)
    return {"champion": bundle["version"], "retrained": decision is not None, "decision": decision,
            "live": {k: v for k, v in live.items() if k != "daily"}, "drift_alerts": [d["feature"] for d in drift if d["status"] == "alert"],
            "published_points": len(payload["forecast"]["time"])}


def publish(df, fc, meta, live, drift, runs, decision, out_dir=None) -> dict:
    out_dir = out_dir or SITE_DATA_DIR
    obs = df.dropna(subset=["actual"]).tail(96)
    hist = registry.registry_history()
    last_decision = decision or next((r for r in reversed(runs) if r["retrained"]), None)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "champion": {"version": meta["version"], "metrics": meta["metrics"], "exported_at": meta["exported_at"],
                     "importance": meta["importance"], "mae_by_horizon": meta["mae_by_horizon"]},
        "now": {"time": obs["time"].iloc[-1].isoformat(), "actual": float(obs["actual"].iloc[-1])},
        "history": {"time": [t.isoformat() for t in obs["time"]], "actual": obs["actual"].round(1).tolist()},
        "forecast": {"time": [pd.Timestamp(t).isoformat() for t in fc["target_time"]], "model": fc["pred"].tolist(),
                     "eso": [None if pd.isna(v) else float(v) for v in fc["eso_forecast"]]},
        "live": live, "drift": drift, "registry": hist, "runs": runs[-48:],
        "last_decision": last_decision if decision else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "console.json").write_text(json.dumps(payload, separators=(",", ":"), default=str))
    return payload


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    a = ap.parse_args()
    out = run(data.synthetic() if a.synthetic else None, force_retrain=a.retrain)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
