"""Train the EV range prediction model used by the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from ev_range_core import build_model_pipeline, ensure_feature_columns

DATA_PATH = Path("Electric_Vehicle_Population_Data (2).csv")
MODEL_PATH = Path("model") / "ev_range_model.joblib"
REFERENCE_YEAR = 2026
RANDOM_SEED = 42


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the EV population CSV in the project root."
        )

    df = pd.read_csv(
        path,
        usecols=["Electric Range", "Model Year", "Electric Vehicle Type"],
    )
    df = df.dropna(subset=["Electric Range", "Model Year", "Electric Vehicle Type"])
    df = df[df["Electric Range"] > 0].copy()
    return df


def build_training_frame(df: pd.DataFrame, random_seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.Series]:
    """Create scenario-based training features from the population dataset."""
    rng = np.random.default_rng(random_seed)

    feature_frame = pd.DataFrame(index=df.index)
    feature_frame["battery_capacity_kwh"] = np.clip(
        df["Electric Range"] / rng.normal(5.4, 0.35, len(df)),
        40,
        110,
    )
    feature_frame["battery_soc_pct"] = rng.integers(35, 101, len(df))
    feature_frame["avg_speed_kmh"] = rng.integers(25, 126, len(df))
    feature_frame["temperature_c"] = rng.integers(-5, 41, len(df))
    feature_frame["ac_on"] = (
        (feature_frame["temperature_c"] >= 27) | (feature_frame["temperature_c"] <= 3)
    ).astype(int)
    feature_frame["terrain"] = rng.choice(
        ["flat", "hilly", "mountainous"],
        size=len(df),
        p=[0.5, 0.35, 0.15],
    )
    feature_frame["vehicle_age_years"] = np.clip(REFERENCE_YEAR - df["Model Year"], 0, 15)
    feature_frame["tire_pressure_psi"] = np.clip(rng.normal(33, 1.7, len(df)), 28, 38)
    feature_frame["payload_kg"] = np.clip(rng.normal(180, 85, len(df)), 0, 500)
    feature_frame["regen_braking"] = (
        df["Electric Vehicle Type"]
        .astype(str)
        .str.contains("Battery Electric Vehicle", case=False, na=False)
    ).astype(int)

    soc_factor = feature_frame["battery_soc_pct"] / 100.0
    speed_penalty = np.clip((feature_frame["avg_speed_kmh"] - 75) * 0.0035, -0.08, 0.22)
    temperature_penalty = np.clip(np.abs(feature_frame["temperature_c"] - 22) * 0.006, 0, 0.18)
    terrain_penalty = feature_frame["terrain"].map(
        {"flat": 0.0, "hilly": 0.07, "mountainous": 0.15}
    )
    payload_penalty = feature_frame["payload_kg"] * 0.00011
    age_penalty = feature_frame["vehicle_age_years"] * 0.01
    tire_bonus = (feature_frame["tire_pressure_psi"] - 33.0) * 0.004
    regen_bonus = feature_frame["regen_braking"] * 0.025
    ac_penalty = feature_frame["ac_on"] * 0.05

    target = (
        df["Electric Range"]
        * soc_factor
        * (
            1
            - speed_penalty
            - temperature_penalty
            - terrain_penalty
            - payload_penalty
            - age_penalty
            - ac_penalty
            + tire_bonus
            + regen_bonus
        )
    )
    noise = rng.normal(0, 7, len(df))
    target = np.clip(target + noise, 30, None)

    return ensure_feature_columns(feature_frame), pd.Series(target, name="estimated_range_km")


def train_and_save_model() -> dict[str, float]:
    df = load_dataset()
    features, target = build_training_frame(df)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    pipeline = build_model_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    metrics = {
        "rows": float(len(features)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return metrics


def main() -> None:
    metrics = train_and_save_model()
    print("Training complete")
    print(f"Rows used: {int(metrics['rows'])}")
    print(f"Validation MAE: {metrics['mae']:.2f} km")
    print(f"Validation R^2: {metrics['r2']:.3f}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
