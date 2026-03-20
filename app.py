"""
EV Range Prediction — Streamlit App  (fixed version)
Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Range Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; }
  h1, h2, h3                  { font-family: 'Syne', sans-serif !important; }
  .stApp                      { background: #0a0e1a; color: #e0e6f0; }
  [data-testid="stSidebar"]   { background: #0f1628; border-right: 1px solid #1e2d4a; }

  .metric-card {
    background: linear-gradient(135deg, #111827 0%, #0f1e35 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,180,255,0.07);
    margin-bottom: 12px;
  }
  .metric-card .val   { font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800; color:#00c6ff; line-height:1.1; }
  .metric-card .unit  { font-size:1rem; color:#6ea8d8; margin-top:4px; }
  .metric-card .lbl   { font-size:0.78rem; color:#4a7090; margin-top:8px; letter-spacing:.08em; text-transform:uppercase; }
  .bar-bg             { background:#1a2540; border-radius:8px; height:14px; overflow:hidden; margin-top:14px; }
  .bar-fill           { height:14px; border-radius:8px; background:linear-gradient(90deg,#0060ff,#00c6ff); }
  .insight-box        { background:#0f1628; border-left:3px solid #00c6ff; border-radius:8px; padding:12px 16px; margin:6px 0; font-size:.9rem; color:#a0bcd8; }
  .placeholder-box    { background:#0f1628; border:1px dashed #1e3a5f; border-radius:16px; padding:60px 40px; text-align:center; color:#2a5070; font-size:1.1rem; }

  .stButton > button {
    width:100%; background:linear-gradient(135deg,#0060ff,#00c6ff);
    color:#fff; border:none; border-radius:10px; padding:14px;
    font-family:'Syne',sans-serif; font-size:1rem; font-weight:700;
    letter-spacing:.04em; cursor:pointer;
  }
  .stButton > button:hover { opacity:.85; }
  label { color:#7a9abf !important; font-size:.84rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None


# ── Model Loader ──────────────────────────────────────────────────────────────
MODEL_PATH = "model/ev_range_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model_loaded = False
if os.path.exists(MODEL_PATH):
    try:
        model = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
else:
    st.warning(
        "⚠️ Model not found at `model/ev_range_model.joblib`.  \n"
        "Run **`python train_model.py`** in your terminal first, then refresh this page."
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.5rem 0 0.5rem;">
  <h1 style="font-size:2.6rem;margin:0;background:linear-gradient(90deg,#00c6ff,#0060ff);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
    ⚡ EV Range Predictor
  </h1>
  <p style="color:#4a7090;margin-top:6px;font-size:.95rem;">
    Estimate how far your electric vehicle will travel based on real-world conditions.
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()


# ── Sidebar Inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔧 Set Conditions")

    st.markdown("**🔋 Battery**")
    battery_capacity = st.slider("Battery Capacity (kWh)", 40, 100, 75)
    battery_soc      = st.slider("State of Charge (%)",    20, 100, 80)

    st.markdown("**🚗 Driving**")
    avg_speed = st.slider("Average Speed (km/h)", 20, 130, 80)
    terrain   = st.selectbox("Terrain Type", ["flat", "hilly", "mountainous"])

    st.markdown("**🌡️ Environment**")
    temperature = st.slider("Temperature (°C)", -10, 45, 22)
    ac_on       = st.toggle("Air Conditioning", value=True)

    st.markdown("**⚙️ Vehicle**")
    car_name      = st.text_input("Car Name", "Generic EV")
    vehicle_age   = st.slider("Vehicle Age (years)", 0, 10, 2)
    tire_pressure = st.slider("Tyre Pressure (PSI)", 28, 38, 33)
    payload       = st.slider("Payload (kg)", 0, 500, 150)
    regen_braking = st.toggle("Regenerative Braking", value=True)

    st.markdown("---")
    predict_btn = st.button("⚡  Predict Range", disabled=not model_loaded)


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_input():
    return pd.DataFrame([{
        "battery_capacity_kwh": battery_capacity,
        "battery_soc_pct":      battery_soc,
        "avg_speed_kmh":        avg_speed,
        "temperature_c":        temperature,
        "ac_on":                int(ac_on),
        "terrain":              terrain,
        "vehicle_age_years":    vehicle_age,
        "tire_pressure_psi":    tire_pressure,
        "payload_kg":           payload,
        "regen_braking":        int(regen_braking),
    }])

def range_label(km):
    if km >= 400: return ("Excellent 🟢", "#00ff88")
    if km >= 250: return ("Good 🟡",      "#ffd700")
    if km >= 100: return ("Low 🟠",       "#ff9500")
    return ("Critical 🔴", "#ff3b30")

def get_insights(km, speed, temp, ac, terrain_val, soc, regen):
    tips = []
    if speed > 100:
        tips.append("🚀 Reducing speed below 100 km/h significantly improves efficiency.")
    if temp < 0:
        tips.append("❄️ Cold weather reduces battery performance — pre-condition the cabin.")
    if temp > 35:
        tips.append("🌡️ Heat increases drain. Park in shade and pre-cool before driving.")
    if ac and km < 300:
        tips.append("💨 Disabling AC could recover ~8% more range.")
    if terrain_val == "mountainous":
        tips.append("⛰️ Mountainous terrain cuts range sharply. Plan charging stops ahead.")
    if soc < 40:
        tips.append("🔋 Low SOC — charge before your trip for a comfortable safety margin.")
    if not regen:
        tips.append("♻️ Enabling regenerative braking can add ~8% extra range.")
    return tips if tips else ["✅ Conditions look optimal for maximum range!"]


# ── Predict on Button Click ───────────────────────────────────────────────────
if predict_btn and model_loaded:
    try:
        inp = build_input()
        km  = float(model.predict(inp)[0])
        st.session_state.result = {
            "car_name": car_name,
            "km":      km,
            "mi":      km * 0.621371,
            "eff":     km / battery_capacity,
            "bar_pct": min(100, (km / 600) * 100),
            "label":   range_label(km),
            "tips":    get_insights(km, avg_speed, temperature, ac_on,
                                    terrain, battery_soc, regen_braking),
            "inputs":  build_input().iloc[0].to_dict(),
        }
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")


# ── Results Display ───────────────────────────────────────────────────────────
if st.session_state.result is None:
    st.markdown("""
    <div class="placeholder-box">
      ⚡ Configure the conditions in the sidebar,<br>then click <strong>Predict Range</strong> to see results.
    </div>
    """, unsafe_allow_html=True)

else:
    r = st.session_state.result
    label_txt, label_color = r["label"]

    st.markdown(f"### 🚗 Results for **{r.get('car_name', 'Generic EV')}**")

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="val">{r['km']:.0f}</div>
          <div class="unit">kilometres</div>
          <div class="lbl">Estimated Range</div>
          <div class="bar-bg">
            <div class="bar-fill" style="width:{r['bar_pct']:.1f}%"></div>
          </div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="val">{r['mi']:.0f}</div>
          <div class="unit">miles</div>
          <div class="lbl">In Miles</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="val">{r['eff']:.2f}</div>
          <div class="unit">km / kWh</div>
          <div class="lbl" style="color:{label_color};">{label_txt}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 💡 Driving Insights")
    for tip in r["tips"]:
        st.markdown(f'<div class="insight-box">{tip}</div>', unsafe_allow_html=True)

    with st.expander("📋 View Input Summary"):
        labels = {
            "battery_capacity_kwh": "Battery Capacity (kWh)",
            "battery_soc_pct":      "State of Charge (%)",
            "avg_speed_kmh":        "Average Speed (km/h)",
            "temperature_c":        "Temperature (°C)",
            "ac_on":                "AC",
            "terrain":              "Terrain",
            "vehicle_age_years":    "Vehicle Age (yrs)",
            "tire_pressure_psi":    "Tyre Pressure (PSI)",
            "payload_kg":           "Payload (kg)",
            "regen_braking":        "Regenerative Braking",
        }
        rows = []
        for k, v in r["inputs"].items():
            display = v
            if k in ("ac_on", "regen_braking"):
                display = "On" if int(v) else "Off"
            elif k == "terrain":
                display = str(v).capitalize()
            rows.append({"Parameter": labels.get(k, k), "Value": display})
        st.table(pd.DataFrame(rows))


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;text-align:center;color:#1e3050;font-size:.78rem;">
  Built with ⚡ Streamlit · Gradient Boosting · For estimation purposes only
</div>
""", unsafe_allow_html=True)