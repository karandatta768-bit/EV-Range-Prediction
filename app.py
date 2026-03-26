"""Streamlit app for EV driving-range estimation."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from ev_range_core import (
    TERRAIN_OPTIONS,
    build_prediction_frame,
    build_summary_rows,
    get_insights,
    range_label,
)

MODEL_PATH = Path("model") / "ev_range_model.joblib"


def configure_page() -> None:
    st.set_page_config(
        page_title="EV Range Predictor",
        page_icon=":zap:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_styles() -> None:
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Manrope:wght@400;500;600;700&display=swap');

          :root {
            --bg: #f3f6fb;
            --panel: rgba(255, 255, 255, 0.78);
            --panel-strong: #ffffff;
            --ink: #0f172a;
            --muted: #61708a;
            --line: rgba(148, 163, 184, 0.22);
            --navy: #10233f;
            --blue: #1769ff;
            --teal: #20b8b0;
            --amber: #f4b942;
            --shadow: 0 22px 55px rgba(15, 23, 42, 0.10);
          }

          html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color: var(--ink); }
          h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--ink); }
          .stApp {
            background:
              radial-gradient(circle at top left, rgba(23, 105, 255, 0.10), transparent 26%),
              radial-gradient(circle at top right, rgba(32, 184, 176, 0.10), transparent 24%),
              linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
          }
          [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1c34 0%, #132846 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
          }
          [data-testid="stSidebar"] * { color: #e7eefb; }
          [data-testid="stSidebar"] label { color: #b6c5e3 !important; }
          [data-testid="stSidebar"] .stMarkdown p { color: #dce7fb; }
          .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1280px; }

          .hero {
            background:
              linear-gradient(135deg, rgba(16, 35, 63, 0.96), rgba(23, 105, 255, 0.88)),
              linear-gradient(120deg, rgba(32, 184, 176, 0.2), transparent);
            border-radius: 28px;
            padding: 30px 34px;
            color: #f8fbff;
            box-shadow: 0 24px 60px rgba(16, 35, 63, 0.22);
            position: relative;
            overflow: hidden;
            margin-bottom: 1.35rem;
          }
          .hero:after {
            content: "";
            position: absolute;
            inset: auto -70px -70px auto;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(255,255,255,0.24), transparent 70%);
          }
          .eyebrow {
            font-size: 0.78rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: rgba(231, 238, 251, 0.72);
            margin-bottom: 0.65rem;
          }
          .hero-title {
            font-size: 3rem;
            line-height: 1.02;
            margin: 0;
            color: #ffffff;
          }
          .hero-subtitle {
            margin-top: 0.8rem;
            max-width: 700px;
            color: rgba(240, 245, 255, 0.82);
            font-size: 1rem;
          }
          .hero-meta {
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            margin-top: 1.15rem;
          }
          .hero-chip {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.14);
            padding: 0.55rem 0.8rem;
            border-radius: 999px;
            font-size: 0.84rem;
            color: #eef4ff;
            backdrop-filter: blur(8px);
          }

          .section-title {
            margin: 1.1rem 0 0.8rem;
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #6a7891;
          }
          .kpi-card, .panel-card, .insight-card, .scenario-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 22px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
          }
          .kpi-card {
            padding: 22px 22px 18px;
            min-height: 168px;
          }
          .kpi-label {
            color: #6a7891;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.74rem;
          }
          .kpi-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 3rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 0.6rem;
            line-height: 1;
          }
          .kpi-unit {
            color: #526079;
            font-size: 0.98rem;
            margin-top: 0.35rem;
          }
          .kpi-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            margin-top: 0.9rem;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            background: rgba(23, 105, 255, 0.08);
            color: #1769ff;
            font-size: 0.83rem;
            font-weight: 600;
          }
          .progress-track {
            margin-top: 1.1rem;
            background: #dbe7ff;
            border-radius: 999px;
            height: 10px;
            overflow: hidden;
          }
          .progress-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #1769ff, #20b8b0);
          }
          .progress-caption {
            color: #72819a;
            font-size: 0.8rem;
            margin-top: 0.55rem;
          }

          .panel-card {
            padding: 24px;
            height: 100%;
          }
          .panel-kicker {
            color: #7d8ba3;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.72rem;
            margin-bottom: 0.55rem;
          }
          .panel-headline {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.55rem;
            color: #0f172a;
            margin-bottom: 0.5rem;
          }
          .panel-copy {
            color: #61708a;
            font-size: 0.95rem;
            line-height: 1.6;
          }
          .mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.85rem;
            margin-top: 1rem;
          }
          .mini-stat {
            background: rgba(15, 23, 42, 0.03);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 16px;
            padding: 0.9rem 1rem;
          }
          .mini-stat-label {
            color: #7d8ba3;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
          }
          .mini-stat-value {
            color: #10233f;
            font-size: 1.15rem;
            font-weight: 700;
            margin-top: 0.35rem;
          }

          .insight-card {
            padding: 18px 18px 16px;
            margin-bottom: 0.8rem;
            display: flex;
            gap: 0.9rem;
            align-items: flex-start;
          }
          .insight-index {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: linear-gradient(135deg, #1769ff, #20b8b0);
            color: white;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.84rem;
            flex-shrink: 0;
          }
          .insight-copy {
            color: #4d5d76;
            line-height: 1.55;
            padding-top: 0.15rem;
          }

          .scenario-card {
            padding: 16px 18px;
            margin-bottom: 0.75rem;
          }
          .scenario-label {
            color: #7d8ba3;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.72rem;
          }
          .scenario-value {
            color: #12233f;
            font-size: 1rem;
            font-weight: 700;
            margin-top: 0.3rem;
          }

          .placeholder-box {
            background: linear-gradient(135deg, rgba(255,255,255,0.82), rgba(255,255,255,0.62));
            border: 1px dashed rgba(23, 105, 255, 0.24);
            border-radius: 24px;
            padding: 76px 40px;
            text-align: center;
            color: #5b6a83;
            box-shadow: var(--shadow);
          }
          .placeholder-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.6rem;
            color: #12233f;
            margin-bottom: 0.55rem;
          }
          .placeholder-copy {
            max-width: 560px;
            margin: 0 auto;
            line-height: 1.65;
          }

          .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #20b8b0, #1769ff);
            color: #ffffff;
            border: none;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1rem;
            font-weight: 700;
            box-shadow: 0 14px 28px rgba(23, 105, 255, 0.22);
          }
          .stButton > button:hover { filter: brightness(1.02); }
          .stExpander {
            background: rgba(255,255,255,0.75);
            border-radius: 18px;
            border: 1px solid var(--line);
          }
          div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            border-radius: 18px;
            overflow: hidden;
          }

          @media (max-width: 900px) {
            .hero { padding: 24px; }
            .hero-title { font-size: 2.25rem; }
            .mini-grid { grid-template-columns: 1fr; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def render_header(model_loaded: bool) -> None:
    status = "Model Ready" if model_loaded else "Model Missing"
    st.markdown(
        f"""
        <section class="hero">
          <div class="eyebrow">EV Decision Support Dashboard</div>
          <h1 class="hero-title">Plan EV trips with clearer, smarter range insights.</h1>
          <p class="hero-subtitle">
            Evaluate battery state, terrain, temperature, speed, and operating conditions in one polished dashboard.
            Adjust the controls in the left panel to simulate a trip and review the predicted range profile instantly.
          </p>
          <div class="hero-meta">
            <span class="hero-chip">Interactive Forecasting</span>
            <span class="hero-chip">Operational Inputs</span>
            <span class="hero-chip">{status}</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model_loaded: bool) -> tuple[dict[str, object], bool]:
    with st.sidebar:
        st.markdown("## Scenario Builder")
        st.caption("Tune the trip assumptions and vehicle condition, then generate a range forecast.")

        st.markdown("**Battery Profile**")
        battery_capacity = st.slider("Battery Capacity (kWh)", 40, 100, 75)
        battery_soc = st.slider("State of Charge (%)", 20, 100, 80)

        st.markdown("**Driving Conditions**")
        avg_speed = st.slider("Average Speed (km/h)", 20, 130, 80)
        terrain = st.selectbox("Terrain Type", TERRAIN_OPTIONS)

        st.markdown("**Ambient Conditions**")
        temperature = st.slider("Temperature (C)", -10, 45, 22)
        ac_on = st.toggle("Air Conditioning", value=True)

        st.markdown("**Vehicle Setup**")
        car_name = st.text_input("Car Name", "Generic EV")
        vehicle_age = st.slider("Vehicle Age (years)", 0, 10, 2)
        tire_pressure = st.slider("Tyre Pressure (PSI)", 28, 38, 33)
        payload = st.slider("Payload (kg)", 0, 500, 150)
        regen_braking = st.toggle("Regenerative Braking", value=True)

        st.markdown("---")
        predict_btn = st.button("Generate Forecast", disabled=not model_loaded)
        if not model_loaded:
            st.caption("Train or restore the model to enable forecasting.")

    inputs = {
        "car_name": car_name,
        "battery_capacity_kwh": battery_capacity,
        "battery_soc_pct": battery_soc,
        "avg_speed_kmh": avg_speed,
        "temperature_c": temperature,
        "ac_on": int(ac_on),
        "terrain": terrain,
        "vehicle_age_years": vehicle_age,
        "tire_pressure_psi": tire_pressure,
        "payload_kg": payload,
        "regen_braking": int(regen_braking),
    }
    return inputs, predict_btn


def run_prediction(model, inputs: dict[str, object]) -> dict[str, object]:
    model_inputs = {key: value for key, value in inputs.items() if key != "car_name"}
    prediction_frame = build_prediction_frame(model_inputs)
    km = float(model.predict(prediction_frame)[0])
    label_text, label_color = range_label(km)

    return {
        "car_name": inputs["car_name"],
        "km": km,
        "mi": km * 0.621371,
        "efficiency": km / float(inputs["battery_capacity_kwh"]),
        "bar_pct": min(100.0, (km / 600.0) * 100.0),
        "label_text": label_text,
        "label_color": label_color,
        "tips": get_insights(
            km=km,
            speed=float(inputs["avg_speed_kmh"]),
            temp=float(inputs["temperature_c"]),
            ac=bool(inputs["ac_on"]),
            terrain=str(inputs["terrain"]),
            soc=float(inputs["battery_soc_pct"]),
            regen=bool(inputs["regen_braking"]),
        ),
        "inputs": model_inputs,
    }


def render_kpi_card(label: str, value: str, unit: str, pill: str) -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-unit">{unit}</div>
      <div class="kpi-pill">{pill}</div>
    </div>
    """


def render_results(result: dict[str, object] | None) -> None:
    if result is None:
        st.markdown(
            """
            <div class="placeholder-box">
              <div class="placeholder-title">Build a scenario to unlock the forecast.</div>
              <div class="placeholder-copy">
                The dashboard will surface estimated range, efficiency, condition-specific guidance,
                and a clean summary of your operating inputs once you generate a forecast.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown('<div class="section-title">Forecast Overview</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3, gap="medium")

    with k1:
        st.markdown(
            render_kpi_card(
                "Estimated Range",
                f"{result['km']:.0f}",
                "kilometres",
                result["label_text"],
            ),
            unsafe_allow_html=True,
        )

    with k2:
        st.markdown(
            render_kpi_card(
                "Equivalent Distance",
                f"{result['mi']:.0f}",
                "miles",
                "Cross-market reference",
            ),
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            render_kpi_card(
                "Energy Efficiency",
                f"{result['efficiency']:.2f}",
                "km per kWh",
                f"Vehicle: {result['car_name']}",
            ),
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.45, 1], gap="large")

    with left:
        st.markdown(
            f"""
            <div class="panel-card">
              <div class="panel-kicker">Executive Summary</div>
              <div class="panel-headline">Scenario outlook: {result['label_text']}</div>
              <div class="panel-copy">
                This forecast reflects the current operating profile for <strong>{result['car_name']}</strong>.
                The prediction combines battery availability, terrain severity, cabin load, and vehicle condition
                into a single trip-range estimate designed for fast scenario comparison.
              </div>
              <div class="mini-grid">
                <div class="mini-stat">
                  <div class="mini-stat-label">Battery State</div>
                  <div class="mini-stat-value">{result['inputs']['battery_soc_pct']}% charged</div>
                </div>
                <div class="mini-stat">
                  <div class="mini-stat-label">Terrain</div>
                  <div class="mini-stat-value">{str(result['inputs']['terrain']).capitalize()}</div>
                </div>
                <div class="mini-stat">
                  <div class="mini-stat-label">Average Speed</div>
                  <div class="mini-stat-value">{result['inputs']['avg_speed_kmh']} km/h</div>
                </div>
                <div class="mini-stat">
                  <div class="mini-stat-label">Ambient Temperature</div>
                  <div class="mini-stat-value">{result['inputs']['temperature_c']} C</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-title">Driving Recommendations</div>', unsafe_allow_html=True)
        for index, tip in enumerate(result["tips"], start=1):
            st.markdown(
                f"""
                <div class="insight-card">
                  <div class="insight-index">{index:02d}</div>
                  <div class="insight-copy">{tip}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="section-title">Scenario Snapshot</div>', unsafe_allow_html=True)
        snapshot_items = [
            ("Battery Capacity", f"{result['inputs']['battery_capacity_kwh']} kWh"),
            ("Vehicle Age", f"{result['inputs']['vehicle_age_years']} years"),
            ("Tyre Pressure", f"{result['inputs']['tire_pressure_psi']} PSI"),
            ("Payload", f"{result['inputs']['payload_kg']} kg"),
            ("Air Conditioning", "On" if result["inputs"]["ac_on"] else "Off"),
            ("Regenerative Braking", "On" if result["inputs"]["regen_braking"] else "Off"),
        ]
        for label, value in snapshot_items:
            st.markdown(
                f"""
                <div class="scenario-card">
                  <div class="scenario-label">{label}</div>
                  <div class="scenario-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="panel-card" style="margin-top:0.9rem;">
              <div class="panel-kicker">Model Use</div>
              <div class="panel-copy">
                Use this dashboard for planning and comparison across scenarios. The forecast is an
                estimation tool and should be paired with real vehicle telemetry for operational decisions.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("View Detailed Input Table"):
        st.table(pd.DataFrame(build_summary_rows(result["inputs"])))


def main() -> None:
    configure_page()
    inject_styles()

    if "result" not in st.session_state:
        st.session_state.result = None

    model = None
    model_loaded = False
    if MODEL_PATH.exists():
        try:
            model = load_model()
            model_loaded = True
        except Exception as exc:
            st.error(f"Failed to load model: {exc}")
    else:
        st.warning(
            "Model not found at `model/ev_range_model.joblib`.\n\n"
            "Run `python train_model.py` in the project root, then refresh the page."
        )

    render_header(model_loaded)
    inputs, predict_btn = render_sidebar(model_loaded)

    if predict_btn and model is not None:
        try:
            st.session_state.result = run_prediction(model, inputs)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    render_results(st.session_state.result)

    st.markdown(
        """
        <div style="margin-top:2rem;text-align:center;color:#6f7f98;font-size:0.8rem;">
          Built with Streamlit and scikit-learn for EV range scenario analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
