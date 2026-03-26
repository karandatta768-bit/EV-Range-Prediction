"""Shared helpers for the EV range prediction app and training script."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURE_COLUMNS = [
    "battery_capacity_kwh",
    "battery_soc_pct",
    "avg_speed_kmh",
    "temperature_c",
    "ac_on",
    "terrain",
    "vehicle_age_years",
    "tire_pressure_psi",
    "payload_kg",
    "regen_braking",
]

NUMERIC_COLUMNS = [
    "battery_capacity_kwh",
    "battery_soc_pct",
    "avg_speed_kmh",
    "temperature_c",
    "vehicle_age_years",
    "tire_pressure_psi",
    "payload_kg",
]

CATEGORICAL_COLUMNS = ["terrain"]
PASSTHROUGH_COLUMNS = ["ac_on", "regen_braking"]
TERRAIN_OPTIONS = ["flat", "hilly", "mountainous"]


def build_prediction_frame(payload: dict) -> pd.DataFrame:
    """Create a one-row prediction frame in the model's expected column order."""
    frame = pd.DataFrame([payload])
    return frame[FEATURE_COLUMNS]


def build_model_pipeline() -> Pipeline:
    """Create the regression pipeline used by both training and inference."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            (
                "cat",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
            ("pass", "passthrough", PASSTHROUGH_COLUMNS),
        ]
    )

    model = GradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=5,
        n_estimators=300,
        random_state=42,
        subsample=0.8,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def range_label(km: float) -> tuple[str, str]:
    """Translate an estimated range into a label and accent color."""
    if km >= 400:
        return ("Excellent", "#00ff88")
    if km >= 250:
        return ("Good", "#ffd700")
    if km >= 100:
        return ("Low", "#ff9500")
    return ("Critical", "#ff3b30")


def get_insights(
    km: float,
    speed: float,
    temp: float,
    ac: bool,
    terrain: str,
    soc: float,
    regen: bool,
) -> list[str]:
    """Generate plain-language driving suggestions for the current scenario."""
    tips: list[str] = []

    if speed > 100:
        tips.append("Reducing speed below 100 km/h will usually improve efficiency.")
    if temp < 0:
        tips.append("Cold weather lowers battery performance, so pre-conditioning helps.")
    if temp > 35:
        tips.append("High temperatures increase energy drain, so shade and pre-cooling help.")
    if ac and km < 300:
        tips.append("Using less cabin cooling may recover a small amount of extra range.")
    if terrain == "mountainous":
        tips.append("Mountain routes cut range faster, so plan charging stops earlier.")
    if soc < 40:
        tips.append("A low state of charge leaves less buffer, so top up before longer trips.")
    if not regen:
        tips.append("Turning regenerative braking on can improve stop-and-go efficiency.")

    if not tips:
        tips.append("These conditions look favorable for a strong driving range.")

    return tips


def build_summary_rows(inputs: dict) -> list[dict[str, object]]:
    """Format model inputs for display in the UI."""
    labels = {
        "battery_capacity_kwh": "Battery Capacity (kWh)",
        "battery_soc_pct": "State of Charge (%)",
        "avg_speed_kmh": "Average Speed (km/h)",
        "temperature_c": "Temperature (C)",
        "ac_on": "Air Conditioning",
        "terrain": "Terrain",
        "vehicle_age_years": "Vehicle Age (years)",
        "tire_pressure_psi": "Tyre Pressure (PSI)",
        "payload_kg": "Payload (kg)",
        "regen_braking": "Regenerative Braking",
    }

    rows: list[dict[str, object]] = []
    for key in FEATURE_COLUMNS:
        value = inputs[key]
        if key in {"ac_on", "regen_braking"}:
            value = "On" if int(value) else "Off"
        elif key == "terrain":
            value = str(value).capitalize()
        rows.append({"Parameter": labels.get(key, key), "Value": value})
    return rows


def ensure_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a training or inference frame contains the expected features."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {', '.join(missing)}")
    return frame[FEATURE_COLUMNS]


def rows_to_frame(rows: Iterable[dict]) -> pd.DataFrame:
    """Convert row dictionaries to a validated feature frame."""
    return ensure_feature_columns(pd.DataFrame(list(rows)))
