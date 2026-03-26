import unittest
from pathlib import Path

import joblib
import pandas as pd

from ev_range_core import (
    FEATURE_COLUMNS,
    build_model_pipeline,
    build_prediction_frame,
    build_summary_rows,
    get_insights,
    range_label,
)


class ProjectTests(unittest.TestCase):
    def test_prediction_frame_respects_feature_order(self):
        payload = {
            "battery_capacity_kwh": 75,
            "battery_soc_pct": 80,
            "avg_speed_kmh": 78,
            "temperature_c": 22,
            "ac_on": 1,
            "terrain": "flat",
            "vehicle_age_years": 2,
            "tire_pressure_psi": 33,
            "payload_kg": 150,
            "regen_braking": 1,
        }

        frame = build_prediction_frame(payload)
        self.assertEqual(frame.columns.tolist(), FEATURE_COLUMNS)
        self.assertEqual(frame.shape, (1, len(FEATURE_COLUMNS)))

    def test_helpers_return_expected_shapes(self):
        self.assertEqual(range_label(410)[0], "Excellent")
        tips = get_insights(220, 105, 30, True, "mountainous", 35, False)
        self.assertTrue(any("Mountain routes" in tip for tip in tips))

        rows = build_summary_rows(
            {
                "battery_capacity_kwh": 75,
                "battery_soc_pct": 80,
                "avg_speed_kmh": 78,
                "temperature_c": 22,
                "ac_on": 1,
                "terrain": "flat",
                "vehicle_age_years": 2,
                "tire_pressure_psi": 33,
                "payload_kg": 150,
                "regen_braking": 1,
            }
        )
        self.assertEqual(len(rows), len(FEATURE_COLUMNS))

    def test_pipeline_can_fit_save_and_load(self):
        frame = pd.DataFrame(
            [
                {
                    "battery_capacity_kwh": 60,
                    "battery_soc_pct": 75,
                    "avg_speed_kmh": 65,
                    "temperature_c": 18,
                    "ac_on": 0,
                    "terrain": "flat",
                    "vehicle_age_years": 1,
                    "tire_pressure_psi": 33,
                    "payload_kg": 120,
                    "regen_braking": 1,
                },
                {
                    "battery_capacity_kwh": 82,
                    "battery_soc_pct": 90,
                    "avg_speed_kmh": 95,
                    "temperature_c": 30,
                    "ac_on": 1,
                    "terrain": "hilly",
                    "vehicle_age_years": 3,
                    "tire_pressure_psi": 32,
                    "payload_kg": 210,
                    "regen_braking": 1,
                },
                {
                    "battery_capacity_kwh": 54,
                    "battery_soc_pct": 55,
                    "avg_speed_kmh": 110,
                    "temperature_c": 8,
                    "ac_on": 1,
                    "terrain": "mountainous",
                    "vehicle_age_years": 7,
                    "tire_pressure_psi": 31,
                    "payload_kg": 300,
                    "regen_braking": 0,
                },
                {
                    "battery_capacity_kwh": 98,
                    "battery_soc_pct": 88,
                    "avg_speed_kmh": 70,
                    "temperature_c": 24,
                    "ac_on": 0,
                    "terrain": "flat",
                    "vehicle_age_years": 1,
                    "tire_pressure_psi": 34,
                    "payload_kg": 180,
                    "regen_braking": 1,
                },
            ]
        )
        target = pd.Series([250, 305, 140, 420])

        pipeline = build_model_pipeline()
        pipeline.fit(frame, target)

        prediction = pipeline.predict(frame.iloc[[0]])[0]
        self.assertGreater(prediction, 0)

        model_path = Path("tests") / ".tmp-model.joblib"
        try:
            joblib.dump(pipeline, model_path)
            loaded = joblib.load(model_path)
            loaded_prediction = loaded.predict(frame.iloc[[1]])[0]
            self.assertGreater(loaded_prediction, 0)
        finally:
            if model_path.exists():
                model_path.unlink()


if __name__ == "__main__":
    unittest.main()
