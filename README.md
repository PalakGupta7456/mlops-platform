# GridOps: an end-to-end MLOps platform around a live forecasting model

[![CI](https://github.com/PalakGupta7456/mlops-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/PalakGupta7456/mlops-platform/actions/workflows/ci.yml)
[![Hourly MLOps loop](https://github.com/PalakGupta7456/mlops-platform/actions/workflows/pipeline.yml/badge.svg)](https://github.com/PalakGupta7456/mlops-platform/actions/workflows/pipeline.yml)
![MLflow](https://img.shields.io/badge/MLflow-registry-0194E2) ![Kubernetes](https://img.shields.io/badge/Kubernetes-kind%20tested-326CE5) ![Terraform](https://img.shields.io/badge/Terraform-AWS%20App%20Runner-7B42BC) ![Prometheus](https://img.shields.io/badge/Prometheus%20%2B%20Grafana-monitored-E6522C) ![Jenkins](https://img.shields.io/badge/Jenkins-pipeline-D24939)

**Live console: https://palakgupta7456.github.io/mlops-platform/**

The model here is deliberately simple: LightGBM forecasting Great Britain's grid carbon intensity 24 hours ahead. **The point of this project is everything around it.** That means continuous training on live data, an MLflow model registry with a **champion/challenger promotion gate**, **drift and live-performance monitoring** that trigger retraining, a serving API instrumented for **Prometheus**, and CI/CD that tests, scans, ships and **rolls the model out to Kubernetes**. It also has infrastructure as code for **AWS**, and a **Jenkins** pipeline.

There is a real, public yardstick: National Grid ESO publishes its own official forecast. Every model version is compared against it on the same half-hours.

![Console](docs/screenshot.png)

## The control loop (hourly, serverless on GitHub Actions)

```mermaid
flowchart TD
    A[Ingest<br/>ESO carbon intensity API + Open-Meteo weather, 5 GB sites] --> B[Upsert observation store<br/>actuals overwrite earlier forecast rows]
    B --> C{Retrain triggers?<br/>daily schedule · PSI drift ≥ 0.25 ·<br/>live MAE > 1.25 × holdout · manual}
    C -- yes --> D[Train challenger<br/>log params/metrics/model to MLflow,<br/>register new version]
    D --> E{Challenger MAE <<br/>champion MAE × 0.98<br/>on the same holdout?}
    E -- yes --> F[Move 'champion' alias<br/>export serving bundle]
    E -- no --> G[Tag version 'rejected']
    F --> H[Trigger CI: rebuild image,<br/>scan, deploy]
    C -- no --> I
    G --> I
    H --> I[Forecast 48 half-hours with champion<br/>log predictions]
    I --> J[Score past predictions vs actuals and vs ESO<br/>drift report · publish console]
```

## Platform components

| Layer | Implementation |
|---|---|
| **Experiment tracking and registry** | MLflow (SQLite backend plus a file artifact store, persisted on the `live-data` branch). Every run logs params, metrics, feature importance, error by horizon and the model, then registers a version tagged `promoted` or `rejected`. Promotion moves the `champion` alias. |
| **Promotion gate** | The challenger and the current champion are scored on *the same* held-out window. A 2% margin stops the champion flip-flopping on noise. Tests cover promotion, rejection and retention. |
| **Retention** | Keeps the newest 6 versions plus the champion and deletes older artifacts. A regression test caught a real bug here: MLflow 3 returns version numbers as integers, so the champion was not recognised and was pruned. |
| **Monitoring** | Prediction log, then live MAE against ESO on identical half-hours, plus a win rate. Feature drift uses PSI and a KS test against the training window. Alerts feed back into the retrain trigger. |
| **Serving** | FastAPI with `/predict`, `/forecast`, `/model`, `/health`, hot `/reload` and `/metrics`. The image is multi-stage and non-root, with a health check. MLflow is not needed at serve time because the champion is exported as a plain bundle. |
| **Observability** | Prometheus metrics: request rate, latency histograms, errors, prediction distribution and model version. Alert rules (validated in CI with `promtool`). A provisioned Grafana dashboard. |
| **Kubernetes** | Deployment with zero-downtime rolling updates, readiness and liveness probes, a read-only root filesystem, non-root user, HPA (2–8 pods on CPU) and a PodDisruptionBudget. Validated with `kubeconform` and **actually deployed to a kind cluster in CI** on every main build. |
| **Cloud IaC** | Terraform for AWS: ECR with scan-on-push and a lifecycle policy, an IAM role, and App Runner with autoscaling and health checks. S3 remote state. `cd-aws.yml` copies the image from GHCR to ECR and applies, behind a `production` environment approval. |
| **CI** | ruff, pytest, a **model quality gate** (must beat ESO), IaC validation, Docker build, **Trivy** image scan, smoke test, push to GHCR, then a kind deploy test. |
| **Jenkins** | The same stages plus a manual `input` approval before production, and an automatic `kubectl rollout undo` if the rollout fails. |

## Run the whole platform locally

```bash
docker compose up --build                       # API :8000 · MLflow :5000 · Prometheus :9090 · Grafana :3000
docker compose run --rm trainer                 # run the control loop against the MLflow server
docker compose --profile load up loadgen        # generate traffic for the Grafana dashboard
```

Without Docker:

```bash
pip install -e ".[train,api,dev]"
python -m gridops.pipeline                      # live data (add --synthetic for offline, --retrain to force)
uvicorn api.main:app --reload
mlflow ui --backend-store-uri sqlite:///state/mlflow.db
pytest
```

## Deploy to Kubernetes / AWS

```bash
kubectl apply -f deploy/k8s/
kubectl -n gridops set image deployment/gridops-api api=ghcr.io/palakgupta7456/gridops:<sha>
```

For AWS, add the `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `TF_STATE_BUCKET` repository secrets, then run **CD to AWS App Runner**.

## Repository layout

```
src/gridops/   data · features · registry (MLflow, gate, retention) · monitoring (PSI, live scoring, triggers) · pipeline
api/           FastAPI + Prometheus
deploy/k8s     Deployment, Service, HPA, PDB
deploy/monitoring  Prometheus config + alert rules, Grafana provisioning + dashboard
infra/terraform    AWS ECR + App Runner
.github/workflows  ci.yml · pipeline.yml (hourly loop) · cd-aws.yml
Jenkinsfile · Dockerfile · docker-compose.yml
```
