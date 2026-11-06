"""Settings for the GridOps MLOps platform."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = Path(os.getenv("GRIDOPS_STATE", ROOT / "state"))      # persisted between runs (live-data branch)
DATA_FILE = STATE_DIR / "observations.parquet"
PRED_LOG = STATE_DIR / "predictions.parquet"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{STATE_DIR / 'mlflow.db'}")
MLFLOW_ARTIFACTS = STATE_DIR / "mlartifacts"
SITE_DATA_DIR = ROOT / "docs" / "data"

MODEL_NAME = "grid-carbon-intensity"
EXPERIMENT = "gridops"
CHAMPION = "champion"

CARBON_API = "https://api.carbonintensity.org.uk"
WEATHER_API = "https://api.open-meteo.com/v1/forecast"
# Weather points that drive GB grid carbon: wind (Scotland, North Sea coast, Wales) and solar/demand (south, midlands)
WEATHER_POINTS = {"scotland": (56.5, -4.2), "north_sea": (54.5, -1.0), "wales": (52.3, -3.8), "south": (51.3, -0.8), "midlands": (52.6, -1.6)}
WEATHER_VARS = ["wind_speed_100m", "shortwave_radiation", "temperature_2m", "cloud_cover"]

HISTORY_DAYS = 90
HORIZONS = list(range(1, 49))          # 48 half-hours = 24 hours ahead
LAGS = [1, 2, 4, 12, 24, 48, 96, 336]  # half-hour steps (336 = one week)
HOLDOUT_DAYS = 10
PROMOTION_MARGIN = 0.02                # challenger must beat champion MAE by 2% to be promoted
DRIFT_PSI_WARN, DRIFT_PSI_ALERT = 0.1, 0.25
KEEP_MODEL_VERSIONS = 6
