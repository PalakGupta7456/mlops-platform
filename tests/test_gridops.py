import importlib
import json

import numpy as np
import pandas as pd
import pytest

END = pd.Timestamp("2026-02-01 12:00")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated state dir + MLflow DB per test."""
    monkeypatch.setenv("GRIDOPS_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'state' / 'mlflow.db'}")
    (tmp_path / "state").mkdir()
    import gridops.config as cfg
    importlib.reload(cfg)
    for mod in ("gridops.data", "gridops.features", "gridops.monitoring", "gridops.registry", "gridops.pipeline"):
        importlib.reload(importlib.import_module(mod))
    from gridops import data, pipeline, registry
    monkeypatch.setattr(pipeline, "SITE_DATA_DIR", tmp_path / "site", raising=False)
    monkeypatch.setattr(registry, "PARAMS", {**registry.PARAMS, "n_estimators": 60})
    return {"tmp": tmp_path, "data": data, "pipeline": pipeline, "registry": registry, "cfg": cfg}


def test_features_have_no_leakage():
    from gridops import data
    from gridops.features import build
    df = data.synthetic(days=10, end=END)
    f = build(df, horizons=[1, 48]).dropna(subset=["target"])
    row = f[f["horizon"] == 48].iloc[100]
    y = df.set_index("time")["actual"]
    assert row["ci_now"] == y[row["time"]]
    assert row["target"] == y[row["time"] + pd.Timedelta(hours=24)]
    assert row["ci_lag_1"] == y[row["time"] - pd.Timedelta(minutes=30)]


def test_psi_detects_shift():
    from gridops.monitoring import psi
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 2000)) < 0.05
    assert psi(a, rng.normal(1.5, 1, 2000)) > 0.25


def test_store_upsert_overwrites_forecast_rows_with_actuals(tmp_path):
    from gridops import data
    path = tmp_path / "obs.parquet"
    df = data.synthetic(days=5, end=END)
    data.update_store(df, path)
    later = df.copy()
    later.loc[later["time"] > END, "actual"] = 123.0
    out = data.update_store(later, path)
    assert out["time"].is_unique and (out.loc[out["time"] > END, "actual"] == 123.0).all()


def test_champion_challenger_lifecycle(env):
    data, pipeline, registry = env["data"], env["pipeline"], env["registry"]
    df = data.synthetic(days=40, end=END)
    first = pipeline.run(df)
    assert first["retrained"] and first["decision"]["promoted"] and first["champion"] == "1"

    # Same data again: challenger is no better -> must be rejected, champion unchanged
    again = pipeline.run(df, force_retrain=True)
    assert again["decision"]["promoted"] is False and again["champion"] == "1"

    hist = registry.registry_history()
    assert [h["status"] for h in hist] == ["rejected", "promoted"]
    assert [h["champion"] for h in hist] == [False, True]
    console = json.loads((env["tmp"] / "site" / "console.json").read_text())
    assert len(console["forecast"]["model"]) == 48 and console["champion"]["version"] == "1"


def test_api_serves_champion_and_metrics(env, monkeypatch):
    env["pipeline"].run(env["data"].synthetic(days=30, end=END))
    from fastapi.testclient import TestClient

    import api.main as api
    monkeypatch.setattr(api, "SERVING_DIR", env["tmp"] / "state" / "serving")
    monkeypatch.setattr(api, "SITE_DATA_DIR", env["tmp"] / "site")
    with TestClient(api.app) as c:
        assert c.get("/health").json()["model_version"] == "1"
        feats = c.get("/model").json()
        assert feats["n_features"] > 20
        bundle = api.STATE["bundle"]
        row = {f: 1.0 for f in bundle["features"]}
        r = c.post("/predict", json={"rows": [row, row]})
        assert r.status_code == 200 and len(r.json()["predictions"]) == 2
        assert c.post("/predict", json={"rows": [{"x": 1.0}]}).status_code == 422
        assert len(c.get("/forecast").json()["forecast"]) == 48
        m = c.get("/metrics").text
        assert "gridops_predictions_total" in m and "gridops_model_version 1.0" in m


def test_retention_never_deletes_the_champion(env, monkeypatch):
    data, pipeline, registry = env["data"], env["pipeline"], env["registry"]
    monkeypatch.setattr(registry, "KEEP_MODEL_VERSIONS", 1)
    df = data.synthetic(days=30, end=END)
    pipeline.run(df)                       # v1 promoted
    pipeline.run(df, force_retrain=True)   # v2 rejected
    pipeline.run(df, force_retrain=True)   # v3 rejected -> v2 pruned, v1 kept as champion
    hist = registry.registry_history()
    assert [h["version"] for h in hist] == ["3", "1"] and hist[1]["champion"]
