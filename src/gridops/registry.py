"""Training with MLflow tracking + Model Registry and a champion/challenger gate.

Every retrain:
  1. trains a LightGBM challenger on all data except the last HOLDOUT_DAYS,
  2. logs params, metrics, feature importance and the model to MLflow,
  3. registers it as a new version of MODEL_NAME,
  4. scores the current champion on the *same* holdout,
  5. promotes the challenger (moves the `champion` alias) only if it beats the
     champion by PROMOTION_MARGIN; otherwise it is tagged `rejected`,
  6. exports the champion to a plain serving bundle (no MLflow needed to serve).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import UTC, datetime

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CHAMPION, EXPERIMENT, HOLDOUT_DAYS, KEEP_MODEL_VERSIONS, MLFLOW_ARTIFACTS, MLFLOW_URI, MODEL_NAME, PROMOTION_MARGIN, STATE_DIR
from .features import PersistenceBlend, build, feature_columns

log = logging.getLogger(__name__)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

PARAMS = {"n_estimators": 700, "learning_rate": 0.04, "num_leaves": 63, "min_child_samples": 40,
          "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0, "random_state": 7, "verbose": -1}


def _mlflow():
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        MLFLOW_ARTIFACTS.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def metrics(y, pred, now) -> dict:
    """Holdout metrics against the fair offline baseline: persistence ("no change from now").
    ESO is compared only in the live log, where its forecast is recorded at our issue time
    (ESO's historical "forecast" field is a last-minute nowcast, not a 24 h-ahead forecast)."""
    err = np.abs(pred - y)
    pers = np.abs(now - y)
    out = {"mae": float(err.mean()), "rmse": float(np.sqrt(((pred - y) ** 2).mean())),
           "mape": float((err / np.maximum(np.abs(y), 1)).mean() * 100),
           "persistence_mae": float(pers.mean()), "skill_vs_persistence": float(1 - err.mean() / pers.mean())}
    return {k: round(v, 4) for k, v in out.items()}


VALIDATION_DAYS = 7


def split(df: pd.DataFrame):
    """train | validation (fits blend weights) | holdout (promotion decision): strictly time-ordered, no overlap."""
    frame = build(df).dropna(subset=["target", "ci_now"])
    cutoff = frame["time"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    vcut = cutoff - pd.Timedelta(days=VALIDATION_DAYS)
    train = frame[frame["target_time"] <= cutoff]
    hold = frame[frame["time"] > cutoff]
    return frame, train, hold, cutoff, vcut


def fit_blend(train: pd.DataFrame, vcut, feats, params) -> PersistenceBlend:
    fit, val = train[train["target_time"] <= vcut], train[train["time"] > vcut]
    probe = lgb.LGBMRegressor(**params).fit(fit[feats], fit["target"] - fit["ci_now"])
    weights = PersistenceBlend.fit_weights(probe.predict(val[feats]), val)
    final = lgb.LGBMRegressor(**params).fit(train[feats], train["target"] - train["ci_now"])
    return PersistenceBlend(final, weights, feats)


def train_and_register(df: pd.DataFrame, params: dict | None = None, reason: str = "scheduled") -> dict:
    mlflow = _mlflow()
    from mlflow import MlflowClient
    client = MlflowClient()
    params = {**PARAMS, **(params or {})}
    frame, train, hold, cutoff, vcut = split(df)
    feats = feature_columns(frame)

    with mlflow.start_run(run_name=f"challenger-{datetime.now(UTC):%Y%m%d-%H%M}") as run:
        model = fit_blend(train, vcut, feats, params)
        pred = model.predict(hold)
        m = metrics(hold["target"].to_numpy(), pred, hold["ci_now"].to_numpy(dtype=float))
        by_h = (hold.assign(e=np.abs(pred - hold["target"])).groupby("horizon")["e"].mean().round(2))
        mlflow.log_params({**params, "n_features": len(feats), "train_rows": len(train), "holdout_rows": len(hold),
                           "holdout_start": str(cutoff), "trigger": reason})
        mlflow.log_metrics(m)
        imp = pd.Series(model.model.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
        mlflow.log_dict({"features": feats, "importance": (imp / imp.sum()).round(4).head(20).to_dict(),
                         "mae_by_horizon": by_h.to_dict(), "blend_weights": model.weights}, "analysis.json")
        info = mlflow.sklearn.log_model(model, name="model", registered_model_name=MODEL_NAME, serialization_format="cloudpickle")
        version = str(info.registered_model_version)

    decision = {"challenger": version, "challenger_mae": m["mae"], "champion": None, "champion_mae": None}
    try:
        champ = client.get_model_version_by_alias(MODEL_NAME, CHAMPION)
        champ_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION}")
        champ_pred = champ_model.predict(hold)
        decision.update(champion=champ.version, champion_mae=round(float(np.abs(champ_pred - hold["target"]).mean()), 4))
    except Exception:  # noqa: BLE001 - first ever run: no champion yet
        champ = None

    promote = champ is None or m["mae"] < decision["champion_mae"] * (1 - PROMOTION_MARGIN)
    client.set_model_version_tag(MODEL_NAME, version, "features", json.dumps(feats))
    client.set_model_version_tag(MODEL_NAME, version, "holdout_mae", str(m["mae"]))
    client.set_model_version_tag(MODEL_NAME, version, "skill_vs_persistence", str(m["skill_vs_persistence"]))
    client.set_model_version_tag(MODEL_NAME, version, "status", "promoted" if promote else "rejected")
    if promote:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION, version)
        export_champion(model, feats, version, m, by_h.to_dict(), (imp / imp.sum()).round(4).head(15).to_dict())
    decision.update(promoted=promote, run_id=run.info.run_id, metrics=m,
                    rule=f"promote if challenger MAE < champion MAE x {1 - PROMOTION_MARGIN:.2f}")
    prune_versions(client)
    log.info("challenger v%s MAE %.2f vs champion %s -> %s", version, m["mae"], decision["champion_mae"], "PROMOTED" if promote else "rejected")
    return decision


def export_champion(model, feats, version, m, by_h, importance, out_dir=None) -> None:
    out_dir = out_dir or STATE_DIR / "serving"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": model.input_columns, "version": version}, out_dir / "model.joblib")
    (out_dir / "model.json").write_text(json.dumps({"name": MODEL_NAME, "version": version, "metrics": m,
                                                    "mae_by_horizon": by_h, "importance": importance,
                                                    "exported_at": datetime.now(UTC).isoformat(timespec="seconds")}, indent=1))


def champion_version(client) -> str | None:
    """Resolve the alias directly: search results do not reliably include aliases."""
    try:
        return str(client.get_model_version_by_alias(MODEL_NAME, CHAMPION).version)
    except Exception:  # noqa: BLE001 - no champion yet
        return None


def prune_versions(client) -> None:
    """Retention: keep the champion and the newest KEEP_MODEL_VERSIONS versions; delete older artifacts."""
    versions = sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version), reverse=True)
    champion = {champion_version(client)}
    for v in versions[KEEP_MODEL_VERSIONS:]:
        if str(v.version) in champion:
            continue
        client.delete_model_version(MODEL_NAME, v.version)
        run_dir = MLFLOW_ARTIFACTS / (v.run_id or "_none_")
        shutil.rmtree(run_dir, ignore_errors=True)


def registry_history() -> list[dict]:
    _mlflow()
    from mlflow import MlflowClient
    client = MlflowClient()
    champ = champion_version(client)
    out = []
    for v in sorted(client.search_model_versions(f"name='{MODEL_NAME}'"), key=lambda v: int(v.version), reverse=True):
        out.append({"version": str(v.version), "created": datetime.fromtimestamp(v.creation_timestamp / 1000, UTC).isoformat(timespec="minutes"),
                    "status": v.tags.get("status"), "mae": float(v.tags.get("holdout_mae", "nan")),
                    "skill": float(v.tags["skill_vs_persistence"]) if v.tags.get("skill_vs_persistence") not in (None, "None") else None,
                    "champion": str(v.version) == champ})
    return out


def load_serving(dir_=None) -> dict:
    return joblib.load((dir_ or STATE_DIR / "serving") / "model.joblib")
