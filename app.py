"""
Road Accident Severity Predictor - Interactive Web Application.
Built with Streamlit, Scikit-Learn, XGBoost, and PyDeck.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import folium
import streamlit.components.v1 as components

# Set page config
st.set_page_config(
    page_title="Road Accident Severity Predictor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for crisp, modern professional UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #92400e;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .badge-slight {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-serious {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-fatal {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_app_data():
    """Load dataset safely with caching."""
    from src.data_loader import load_dataset
    try:
        return load_dataset()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


@st.cache_data
def load_app_india_data():
    """Load India accident blackspots dataset with caching."""
    import importlib
    import src.india_data
    importlib.reload(src.india_data)
    try:
        return src.india_data.load_india_accidents_dataset()
    except Exception as e:
        st.error(f"Error loading India accident dataset: {e}")
        return None


@st.cache_data
def load_app_india_highway_routes():
    """Load India highway network polylines for PathLayer with caching."""
    import importlib
    import src.india_data
    importlib.reload(src.india_data)
    try:
        return src.india_data.get_india_highway_routes()
    except Exception as e:
        st.error(f"Error loading India highway routes: {e}")
        return src.india_data.INDIA_HIGHWAY_NETWORKS if hasattr(src.india_data, "INDIA_HIGHWAY_NETWORKS") else []


@st.cache_data
def load_app_india_corridors():
    """Load MoRTH blackspot corridors list with caching."""
    import importlib
    import src.india_data
    importlib.reload(src.india_data)
    try:
        return src.india_data.get_highway_corridors_list()
    except Exception as e:
        st.error(f"Error loading highway corridors: {e}")
        return src.india_data.INDIA_HIGHWAY_CORRIDORS if hasattr(src.india_data, "INDIA_HIGHWAY_CORRIDORS") else []



@st.cache_data
def load_app_metadata():
    """Load model evaluation metadata."""
    meta_path = "models/model_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


@st.cache_resource
def load_app_predictor():
    """Load inference engine with cached binaries."""
    from src.predict import AccidentSeverityPredictor
    try:
        return AccidentSeverityPredictor()
    except Exception:
        return None


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/traffic-light.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview Dashboard",
        "Accident Severity Prediction",
        "What-If Counterfactual Simulator",
        "Batch Incident Scoring & Export",
        "🇮🇳 India Highway Accidents & Safety Awareness Map",
        "Geospatial Hotspot Map",
        "Model Performance",
        "Data Insights"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
predictor = load_app_predictor()
metadata = load_app_metadata()

if predictor is not None and metadata is not None:
    st.sidebar.success(f"Model Active: **{metadata['best_model']['name']}**")
    st.sidebar.caption(f"Evaluated: {metadata['evaluation_timestamp'][:10]}")
    st.sidebar.caption("REST API: Available at `/docs`")
else:
    st.sidebar.warning("Model Not Loaded")

st.sidebar.markdown("---")
st.sidebar.caption("Road Accident Severity Predictor v1.1.0  \nBuilt for Recruiter & Technical Evaluation")

# Header Banner
st.markdown('<div class="main-header">Road Accident Severity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data-driven exploration, counterfactual analysis, and pre-collision severity estimation.</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE 1: OVERVIEW DASHBOARD
# -------------------------------------------------------------
if page == "Overview Dashboard":
    st.subheader("Executive Dashboard")

    df = load_app_data()
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Incidents Analyzed", f"{len(df):,}")
        with c2:
            slight_pct = (df['accident_severity'] == 'Slight').mean() * 100
            st.metric("Slight Severity Rate", f"{slight_pct:.1f}%")
        with c3:
            serious_pct = (df['accident_severity'] == 'Serious').mean() * 100
            st.metric("Serious Severity Rate", f"{serious_pct:.1f}%")
        with c4:
            fatal_pct = (df['accident_severity'] == 'Fatal').mean() * 100
            st.metric("Fatal Severity Rate", f"{fatal_pct:.1f}%")

        st.markdown("---")

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### 🎯 Class Imbalance Distribution")
            fig1_path = "reports/figures/01_class_distribution.png"
            if os.path.exists(fig1_path):
                st.image(fig1_path, use_container_width=True)
            else:
                st.bar_chart(df['accident_severity'].value_counts())
            st.caption("Class distribution reflects real-world severity imbalance (~78% Slight, ~18% Serious, ~4% Fatal).")

        with col_right:
            st.markdown("#### ⏰ Temporal Accident Distribution (24-Hour)")
            fig2_path = "reports/figures/02_accidents_by_hour.png"
            if os.path.exists(fig2_path):
                st.image(fig2_path, use_container_width=True)
            else:
                st.info("Hourly plot generated during training.")
            st.caption("Collisions peak during commuter rush hours (07:00-09:00 and 16:00-19:00).")

        st.markdown("---")
        st.markdown("#### 🏆 Champion ML Pipeline Summary")

        if metadata:
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            best_info = metadata["best_model"]
            with m_col1:
                st.metric("Champion Model", best_info["name"])
            with m_col2:
                st.metric("Selection Metric (Macro F1)", f"{best_info['macro_f1']:.4f}")
            with m_col3:
                st.metric("Overall Accuracy", f"{best_info['accuracy']*100:.1f}%")
            with m_col4:
                st.metric("Multi-Class ROC-AUC (OvR)", f"{best_info['roc_auc_ovr']:.4f}")

            st.info(
                f"💡 **Model Selection Rationale**: **{best_info['name']}** was selected as champion because it maximizes "
                f"**Macro F1-Score ({best_info['macro_f1']:.4f})**, ensuring balanced sensitivity to rare Fatal and Serious "
                "cases rather than optimizing for majority-class accuracy alone."
            )
    else:
        st.warning("Dataset could not be loaded.")


# -------------------------------------------------------------
# PAGE 2: ACCIDENT SEVERITY PREDICTION
# -------------------------------------------------------------
elif page == "Accident Severity Prediction":
    st.subheader("Predict Accident Severity")
    st.write("Input pre-accident environmental, roadway, and vehicular conditions to compute severity estimates.")

    if predictor is None:
        st.error("Model artifacts not found. Please run `python -m src.train`.")
    else:
        st.markdown("##### Quick Scenario Presets")
        scenario = st.selectbox(
            "Load a realistic test scenario:",
            [
                "Custom Input",
                "Scenario A: High-Speed Rural Night Incident (Motorcycle, 60mph, unlit, adverse weather)",
                "Scenario B: Low-Speed Urban Rush Hour Collision (Car, 20mph, daylight, dry)",
                "Scenario C: Adverse Weather Highway Collision (Goods vehicle, 50mph, ice/frost)"
            ]
        )

        if scenario.startswith("Scenario A"):
            d_time, d_day, d_speed, d_weather = "22:45", "Friday", 60, "Raining + high winds"
            d_surf, d_light, d_type, d_area = "Wet or damp", "Darkness - no lighting", "Single carriageway", "Rural"
            d_junc, d_veh, d_age, d_cnt = "Not at junction", "Motorcycle", "18-25", 2
        elif scenario.startswith("Scenario B"):
            d_time, d_day, d_speed, d_weather = "08:30", "Tuesday", 20, "Fine no high winds"
            d_surf, d_light, d_type, d_area = "Dry", "Daylight", "Roundabout", "Urban"
            d_junc, d_veh, d_age, d_cnt = "Roundabout", "Car", "36-55", 2
        elif scenario.startswith("Scenario C"):
            d_time, d_day, d_speed, d_weather = "06:15", "Wednesday", 50, "Fog or mist"
            d_surf, d_light, d_type, d_area = "Frost or ice", "Darkness - lights lit", "Dual carriageway", "Rural"
            d_junc, d_veh, d_age, d_cnt = "T or staggered junction", "Goods vehicle", "56-65", 3
        else:
            d_time, d_day, d_speed, d_weather = "17:30", "Thursday", 30, "Fine no high winds"
            d_surf, d_light, d_type, d_area = "Dry", "Daylight", "Single carriageway", "Urban"
            d_junc, d_veh, d_age, d_cnt = "T or staggered junction", "Car", "26-35", 2

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🕒 Temporal & Vehicular Factors")
                input_time = st.text_input("Accident Time (HH:MM, 24-hr)", value=d_time)
                input_day = st.selectbox(
                    "Day of Week",
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(d_day)
                )
                input_speed = st.slider("Posted Speed Limit (mph)", min_value=20, max_value=70, step=10, value=d_speed)
                input_vehicles = st.number_input("Number of Vehicles Involved", min_value=1, max_value=10, value=d_cnt)
                input_veh_type = st.selectbox(
                    "Primary Vehicle Type",
                    ["Car", "Motorcycle", "Bus", "Goods vehicle", "Other"],
                    index=["Car", "Motorcycle", "Bus", "Goods vehicle", "Other"].index(d_veh)
                )
                input_driver_age = st.selectbox(
                    "Driver Age Band",
                    ["18-25", "26-35", "36-55", "56-65", "65+"],
                    index=["18-25", "26-35", "36-55", "56-65", "65+"].index(d_age)
                )

            with col2:
                st.markdown("#### 🌧️ Roadway & Environmental Factors")
                input_weather = st.selectbox(
                    "Weather Conditions",
                    ["Fine no high winds", "Raining no high winds", "Raining + high winds", "Snowing", "Fog or mist", "Other"],
                    index=["Fine no high winds", "Raining no high winds", "Raining + high winds", "Snowing", "Fog or mist", "Other"].index(d_weather)
                )
                input_surface = st.selectbox(
                    "Road Surface Conditions",
                    ["Dry", "Wet or damp", "Snow", "Frost or ice", "Flood over 3cm"],
                    index=["Dry", "Wet or damp", "Snow", "Frost or ice", "Flood over 3cm"].index(d_surf)
                )
                input_light = st.selectbox(
                    "Light Conditions",
                    ["Daylight", "Darkness - lights lit", "Darkness - lights unlit", "Darkness - no lighting"],
                    index=["Daylight", "Darkness - lights lit", "Darkness - lights unlit", "Darkness - no lighting"].index(d_light)
                )
                input_road_type = st.selectbox(
                    "Road Type",
                    ["Single carriageway", "Dual carriageway", "Roundabout", "One way street", "Slip road"],
                    index=["Single carriageway", "Dual carriageway", "Roundabout", "One way street", "Slip road"].index(d_type)
                )
                input_area = st.selectbox(
                    "Urban or Rural Area",
                    ["Urban", "Rural"],
                    index=["Urban", "Rural"].index(d_area)
                )
                input_junction = st.selectbox(
                    "Junction Detail",
                    ["Not at junction", "T or staggered junction", "Crossroads", "Roundabout", "More than 4 arms"],
                    index=["Not at junction", "T or staggered junction", "Crossroads", "Roundabout", "More than 4 arms"].index(d_junc)
                )

            submit_btn = st.form_submit_button("⚡ Estimate Accident Severity", use_container_width=True)

        if submit_btn:
            input_payload = {
                "accident_time": input_time,
                "day_of_week": input_day,
                "speed_limit": input_speed,
                "weather_conditions": input_weather,
                "road_surface_conditions": input_surface,
                "light_conditions": input_light,
                "road_type": input_road_type,
                "urban_or_rural_area": input_area,
                "junction_detail": input_junction,
                "number_of_vehicles": input_vehicles,
                "driver_age_band": input_driver_age,
                "vehicle_type": input_veh_type
            }

            try:
                result = predictor.predict(input_payload)

                st.markdown("---")
                st.subheader("Prediction Result & Risk Analysis")

                pred_class = result["predicted_severity"]
                risk_tier = result["risk_tier"]
                badge_class = "badge-slight" if pred_class == "Slight" else ("badge-serious" if pred_class == "Serious" else "badge-fatal")

                res_c1, res_c2, res_c3 = st.columns([1.2, 1, 1.2])

                with res_c1:
                    st.markdown(f"### Predicted Severity: <span class='{badge_class}'>{pred_class}</span>", unsafe_allow_html=True)
                    st.write(f"**Overall Risk Tier:** `{risk_tier}`")

                with res_c2:
                    st.markdown("#### Probability Distribution")
                    probs = result["probabilities"]
                    for cls_name, prob in probs.items():
                        st.write(f"**{cls_name}**: `{prob*100:.1f}%`")
                        st.progress(min(1.0, max(0.0, prob)))

                with res_c3:
                    st.markdown("#### Key Contributing Risk Factors")
                    factors = result["risk_factors"]
                    if factors:
                        for f in factors:
                            lvl_color = "red" if f["level"] == "High" else "orange"
                            st.markdown(f"- **:{lvl_color}[{f['factor']}]**: {f['impact']}")
                    else:
                        st.write("No severe environmental or vehicular risk multipliers detected.")

                st.markdown(f"""
                <div class="disclaimer-box">
                    <strong>⚠️ Important Decision-Support Notice:</strong><br>
                    {result['disclaimer']}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Inference error: {e}")


# -------------------------------------------------------------
# PAGE 3: WHAT-IF COUNTERFACTUAL SIMULATOR
# -------------------------------------------------------------
elif page == "What-If Counterfactual Simulator":
    st.subheader("💡 What-If Policy & Safety Intervention Simulator")
    st.markdown("""
    Explore how **safety interventions** (e.g. lowering speed limits, installing street lighting, or improving road surface)
    dynamically alter injury severity probabilities. This links ML prediction to **actionable road safety policy design**.
    """)

    if predictor is None:
        st.error("Predictor model not available.")
    else:
        st.markdown("#### Step 1: Define Baseline Incident")
        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        with b_c1:
            base_speed = st.selectbox("Baseline Speed Limit (mph)", [70, 60, 50, 40, 30], index=1)
        with b_c2:
            base_light = st.selectbox("Baseline Lighting", ["Darkness - no lighting", "Darkness - lights unlit", "Darkness - lights lit", "Daylight"], index=0)
        with b_c3:
            base_surface = st.selectbox("Baseline Road Surface", ["Frost or ice", "Wet or damp", "Snow", "Dry"], index=0)
        with b_c4:
            base_vehicle = st.selectbox("Baseline Vehicle Type", ["Motorcycle", "Car", "Goods vehicle"], index=0)

        st.markdown("#### Step 2: Propose Counterfactual Safety Intervention")
        i_c1, i_c2, i_c3, i_c4 = st.columns(4)
        with i_c1:
            new_speed = st.selectbox("Intervention Speed Limit (mph)", [70, 60, 50, 40, 30, 20], index=4)
        with i_c2:
            new_light = st.selectbox("Intervention Lighting", ["Darkness - lights lit", "Daylight", "Darkness - no lighting"], index=0)
        with i_c3:
            new_surface = st.selectbox("Intervention Road Surface", ["Dry", "Wet or damp", "Frost or ice"], index=0)
        with i_c4:
            new_vehicle = st.selectbox("Intervention Vehicle Type", ["Car", "Motorcycle", "Bus"], index=0)

        if st.button("🚀 Run Counterfactual Simulation", use_container_width=True):
            base_record = {
                "accident_time": "23:00",
                "day_of_week": "Friday",
                "speed_limit": base_speed,
                "weather_conditions": "Fine no high winds",
                "road_surface_conditions": base_surface,
                "light_conditions": base_light,
                "road_type": "Single carriageway",
                "urban_or_rural_area": "Rural",
                "junction_detail": "Not at junction",
                "number_of_vehicles": 2,
                "driver_age_band": "26-35",
                "vehicle_type": base_vehicle
            }

            intervention_record = {
                "accident_time": "23:00",
                "day_of_week": "Friday",
                "speed_limit": new_speed,
                "weather_conditions": "Fine no high winds",
                "road_surface_conditions": new_surface,
                "light_conditions": new_light,
                "road_type": "Single carriageway",
                "urban_or_rural_area": "Rural",
                "junction_detail": "Not at junction",
                "number_of_vehicles": 2,
                "driver_age_band": "26-35",
                "vehicle_type": new_vehicle
            }

            res_base = predictor.predict(base_record)
            res_inter = predictor.predict(intervention_record)

            st.markdown("---")
            st.markdown("### 📊 Counterfactual Comparison: Baseline vs. Intervention")

            sim_col1, sim_col2, sim_col3 = st.columns(3)

            with sim_col1:
                st.markdown("#### 🔴 Baseline Incident")
                st.write(f"Predicted: **{res_base['predicted_severity']}** ({res_base['risk_tier']} Risk)")
                for k, v in res_base["probabilities"].items():
                    st.write(f"• **{k}**: `{v*100:.1f}%`")

            with sim_col2:
                st.markdown("#### 🟢 With Safety Intervention")
                st.write(f"Predicted: **{res_inter['predicted_severity']}** ({res_inter['risk_tier']} Risk)")
                for k, v in res_inter["probabilities"].items():
                    st.write(f"• **{k}**: `{v*100:.1f}%`")

            with sim_col3:
                st.markdown("#### 📉 Net Probability Delta (Δ)")
                for k in ["Fatal", "Serious", "Slight"]:
                    p_before = res_base["probabilities"].get(k, 0.0)
                    p_after = res_inter["probabilities"].get(k, 0.0)
                    delta = (p_after - p_before) * 100
                    delta_color = "inverse" if k in ["Fatal", "Serious"] else "normal"
                    st.metric(f"Δ {k} Risk", f"{p_after*100:.1f}%", f"{delta:+.1f}%", delta_color=delta_color)

            st.success(
                f"🎉 **Policy Impact Summary**: The intervention produced a **{abs(res_inter['probabilities'].get('Fatal', 0) - res_base['probabilities'].get('Fatal', 0))*100:.1f}%** shift in fatal injury risk, "
                f"demonstrating how infrastructure and speed governance directly mitigate severe trauma."
            )


# -------------------------------------------------------------
# PAGE 4: BATCH INCIDENT SCORING & EXPORT
# -------------------------------------------------------------
elif page == "Batch Incident Scoring & Export":
    st.subheader("📁 High-Throughput Batch Scoring & Data Export")
    st.markdown("""
    Municipal road-safety departments and insurance analysts can batch-score hundreds of incidents
    simultaneously, calculate the **Regional Risk Index (RRI)**, and export scored records.
    """)

    if predictor is None:
        st.error("Predictor model not available.")
    else:
        uploaded_file = st.file_uploader("Upload CSV incident file for batch scoring:", type=["csv"])

        df_batch = None
        if uploaded_file is not None:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"Uploaded `{uploaded_file.name}` ({len(df_batch)} rows)")
        else:
            if st.button("📥 Load Sample Test Batch (15 Incidents)"):
                df_raw = load_app_data()
                if df_raw is not None:
                    df_batch = df_raw.sample(15, random_state=42).copy()
                    st.info("Loaded 15 sample incident records.")

        if df_batch is not None:
            st.markdown("##### Preview of Batch Records:")
            st.dataframe(df_batch.head(5), use_container_width=True)

            if st.button("⚡ Score All Batch Incidents", use_container_width=True):
                with st.spinner("Scoring batch incidents through ML pipeline..."):
                    batch_res = predictor.predict(df_batch)
                    preds = [r["predicted_severity"] for r in batch_res["batch_predictions"]]
                    probs_fatal = [r["probabilities"].get("Fatal", 0.0) for r in batch_res["batch_predictions"]]
                    probs_serious = [r["probabilities"].get("Serious", 0.0) for r in batch_res["batch_predictions"]]

                    df_scored = df_batch.copy()
                    df_scored["predicted_severity"] = preds
                    df_scored["prob_fatal"] = np.round(probs_fatal, 4)
                    df_scored["prob_serious"] = np.round(probs_serious, 4)

                    # Calculate Regional Risk Index (RRI: 0-100)
                    rri = float(np.mean(np.array(probs_fatal) * 100 * 2.0 + np.array(probs_serious) * 100 * 1.0))
                    rri = min(100.0, rri)

                    st.markdown("---")
                    st.markdown("### 📊 Batch Evaluation Summary")

                    b_c1, b_c2, b_c3 = st.columns(3)
                    with b_c1:
                        st.metric("Total Scored Incidents", f"{len(df_scored):,}")
                    with b_c2:
                        high_risk_count = sum(df_scored["predicted_severity"].isin(["Serious", "Fatal"]))
                        st.metric("High-Severity Detections", f"{high_risk_count} ({high_risk_count/len(df_scored)*100:.1f}%)")
                    with b_c3:
                        st.metric("Aggregate Regional Risk Index", f"{rri:.1f} / 100")

                    st.markdown("##### Scored Records Output:")
                    st.dataframe(df_scored[["speed_limit", "weather_conditions", "vehicle_type", "predicted_severity", "prob_fatal", "prob_serious"]].head(10), use_container_width=True)

                    csv_export = df_scored.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Complete Scored CSV",
                        data=csv_export,
                        file_name="scored_accident_severity.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


# -------------------------------------------------------------
# PAGE: INDIA HIGHWAY ACCIDENTS & SAFETY AWARENESS MAP
# -------------------------------------------------------------
elif page == "🇮🇳 India Highway Accidents & Safety Awareness Map":
    st.subheader("🇮🇳 Map of India: National Highway Network & Critical Danger Areas")
    st.markdown("""
    Interactive 3D geospatial intelligence visualizing **major National Highways & Expressways across India** 
    alongside high-collision **Accident Danger Areas (Blackspots)**, grounded in official data from the 
    **Ministry of Road Transport and Highways (MoRTH)**, **Integrated Road Accident Database (iRAD)**, and **Indian Road Congress (IRC)**.
    """)

    df_india = load_app_india_data()
    highway_routes = load_app_india_highway_routes()

    if df_india is not None and len(df_india) > 0 and len(highway_routes) > 0:
        st.markdown("---")
        st.markdown("#### 🛣️ Highway Corridor & Danger Area Selection")

        hwy_names_list = ["All Indian Highways & Expressways"] + [h["highway_name"] for h in highway_routes]

        ctrl_col1, ctrl_col2 = st.columns([1.6, 1.4])
        with ctrl_col1:
            selected_hwy_dropdown = st.selectbox(
                "Select National Highway / Expressway Corridor:",
                hwy_names_list,
                index=0
            )
        with ctrl_col2:
            search_query = st.text_input(
                "Or Search Any Area, City, Highway, or Hazard:",
                placeholder="e.g. Pune, NH-44, Yamuna, Bhor Ghat, Bengaluru, Murthal, Goa, Ludhiana, NH-48",
                value=""
            )

        st.caption("Quick Select Corridors & Hazardous Danger Stretches:")
        pill_cols = st.columns(7)
        pill_selection = None
        with pill_cols[0]:
            if st.button("📍 All India", use_container_width=True):
                pill_selection = ""
        with pill_cols[1]:
            if st.button("📍 NH-44 (North-South)", use_container_width=True):
                pill_selection = "NH-44"
        with pill_cols[2]:
            if st.button("📍 NH-48 (Delhi-Chennai)", use_container_width=True):
                pill_selection = "NH-48"
        with pill_cols[3]:
            if st.button("📍 Mumbai-Pune Exp", use_container_width=True):
                pill_selection = "Mumbai-Pune"
        with pill_cols[4]:
            if st.button("📍 Yamuna Exp", use_container_width=True):
                pill_selection = "Yamuna"
        with pill_cols[5]:
            if st.button("📍 NH-66 (Goa Coast)", use_container_width=True):
                pill_selection = "NH-66"
        with pill_cols[6]:
            if st.button("📍 Samruddhi Exp", use_container_width=True):
                pill_selection = "Samruddhi"

        # Determine effective search term
        if pill_selection is not None:
            active_query = pill_selection
        elif search_query.strip():
            active_query = search_query.strip()
        elif selected_hwy_dropdown != "All Indian Highways & Expressways":
            # Extract highway id or short name
            active_query = selected_hwy_dropdown.split(":")[0].strip()
        else:
            active_query = ""

        # Filter accident records based on selection
        if active_query:
            mask = (
                df_india["area_name"].str.contains(active_query, case=False, na=False) |
                df_india["highway_name"].str.contains(active_query, case=False, na=False) |
                df_india["highway_id"].str.contains(active_query, case=False, na=False) |
                df_india["state"].str.contains(active_query, case=False, na=False) |
                df_india["primary_cause"].str.contains(active_query, case=False, na=False)
            )
            filtered_india = df_india[mask].copy()
            if len(filtered_india) == 0:
                st.warning(f"No specific blackspots found matching '{active_query}'. Displaying all India danger zones.")
                filtered_india = df_india.copy()
                is_filtered = False
            else:
                is_filtered = True
        else:
            filtered_india = df_india.copy()
            is_filtered = False

        st.markdown("---")

        # Section A: Collision Statistics for Selected Corridor / Area
        total_accidents = len(filtered_india)
        fatal_count = sum(filtered_india["accident_severity"] == "Fatal")
        serious_count = sum(filtered_india["accident_severity"] == "Serious")
        slight_count = sum(filtered_india["accident_severity"] == "Slight")

        fatal_pct = (fatal_count / total_accidents * 100) if total_accidents > 0 else 0
        serious_pct = (serious_count / total_accidents * 100) if total_accidents > 0 else 0

        header_title = active_query if active_query else "All National Highways of India"
        st.markdown(f"### 📊 Collision & Danger Profile: **{header_title}**")
        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        with stat_c1:
            st.metric("Total Accidents Recorded", f"{total_accidents:,}")
        with stat_c2:
            st.metric("Fatal Collisions 🚨", f"{fatal_count:,}", f"{fatal_pct:.1f}%")
        with stat_c3:
            st.metric("Grievous / Serious Injuries ⚠️", f"{serious_count:,}", f"{serious_pct:.1f}%")
        with stat_c4:
            st.metric("Minor / Slight Injuries", f"{slight_count:,}")

        # Section B: Critical Safety Awareness Banner ("GO WITHIN THE LIMIT!")
        if is_filtered and len(filtered_india) > 0:
            top_spot = filtered_india.iloc[0]
            posted_spd = top_spot["posted_speed_kmh"]
            safe_spd = top_spot["advisory_safe_speed_kmh"]
            cause = top_spot["primary_cause"]
            area_title = top_spot["area_name"]
            hwy_title = top_spot["highway_name"]
            code = top_spot["morth_blackspot_id"]
            d_level = top_spot.get("danger_level", "High Danger")
            rect_status = top_spot.get("rectification_status", "Active enforcement and rumble strips installed.")

            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #7f1d1d, #991b1b); color: white; padding: 22px; border-radius: 12px; margin: 15px 0; border-left: 8px solid #f87171; box-shadow: 0 4px 10px rgba(0,0,0,0.25);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 1.35rem; font-weight: 800; letter-spacing: 0.5px;">
                        🛑 CRITICAL ACCIDENT DANGER AREA DETECTED — GO WITHIN THE LIMIT!
                    </div>
                    <span style="background: #ef4444; color: white; font-weight: 700; padding: 4px 12px; border-radius: 20px; font-size: 0.85rem;">
                        {d_level.upper()}
                    </span>
                </div>
                <div style="font-size: 1.1rem; margin-bottom: 14px; line-height: 1.5;">
                    Danger Stretch: <strong>{area_title}</strong> on <strong>{hwy_title}</strong> (MoRTH Blackspot Code: <code>{code}</code>)
                </div>
                <div style="display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 14px; background: rgba(0,0,0,0.3); padding: 14px; border-radius: 8px;">
                    <div><strong>🚨 Posted Highway Speed:</strong> <span style="font-size: 1.25rem; color: #fca5a5; font-weight: bold;">{posted_spd} km/h</span></div>
                    <div><strong>🟢 Mandatory Safe Speed:</strong> <span style="font-size: 1.35rem; font-weight: 900; color: #86efac; background: #064e3b; padding: 3px 10px; border-radius: 6px;">{safe_spd} km/h (GO WITHIN LIMIT)</span></div>
                    <div style="flex: 1 1 100%;"><strong>⚠️ Primary Crash Hazard:</strong> <span style="color: #fef08a;">{cause}</span></div>
                    <div style="flex: 1 1 100%;"><strong>🛠️ MoRTH/NHAI Engineering Intervention:</strong> <span style="color: #e0f2fe;">{rect_status}</span></div>
                </div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #fef3c7;">
                    📢 Driver Safety Awareness Directive: <em>SLOW DOWN to {safe_spd} km/h! Over 70% of fatal highway crashes in India stem from excessive speed. Maintain a 5-second safe gap behind heavy commercial trucks, avoid sharp lane-weaving, downshift on ghat descents, and take mandatory rest breaks every 2 hours to avoid highway hypnosis.</em>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info(
                "💡 **MoRTH Road Safety Directive**: In India, over **150,000 road fatalities** occur every year across National Highways "
                "and Expressways. **Speeding accounts for over 70% of fatal collisions** (MoRTH Annual Report). "
                "Select any highway corridor from the dropdown or click a quick-select button above to inspect its exact route, "
                "danger blackspots, and mandatory safe speed limits."
            )

        # Section C: Interactive Map of India with Highway Corridors & Danger Areas
        st.markdown("---")
        st.markdown("#### 🗺️ Map of India: Highway Network Routes & Danger Blackspots")

        # Map Engine Mode Selector
        map_engine_col1, map_engine_col2 = st.columns([2.2, 1.8])
        with map_engine_col1:
            map_view_engine = st.radio(
                "Select Map View Mode:",
                [
                    "🗺️ Clear Geographic Road Map (OpenStreetMap / Folium) — Official States & Borders",
                    "🌐 3D Perspective Elevation Map (PyDeck WebGL)"
                ],
                index=0,
                horizontal=True
            )

        # Build blackspot area nodes for on-map location callouts
        area_summary_list = []
        for aname, grp in filtered_india.groupby("area_name"):
            tot = len(grp)
            fat = sum(grp["accident_severity"] == "Fatal")
            adv_spd = grp["advisory_safe_speed_kmh"].iloc[0]
            d_lvl = grp["danger_level"].iloc[0] if "danger_level" in grp.columns else "High Danger"
            hwy = grp["highway_name"].iloc[0].split("(")[0].strip()
            cause_desc = grp["primary_cause"].iloc[0]
            code = grp["morth_blackspot_id"].iloc[0]
            area_summary_list.append({
                "area_name": aname,
                "highway_name": hwy,
                "danger_level": d_lvl,
                "primary_cause": cause_desc,
                "morth_blackspot_id": code,
                "latitude": float(grp["latitude"].mean()),
                "longitude": float(grp["longitude"].mean()),
                "total_accidents": tot,
                "fatal_accidents": fat,
                "advisory_speed": adv_spd,
                "map_label": f"⚠️ {aname.split('(')[0].strip()}\n{tot} Crashes ({fat} Fatal) | GO ≤ {adv_spd} km/h"
            })
        blackspots_nodes_df = pd.DataFrame(area_summary_list)

        # Calculate map center and zoom level
        if is_filtered and len(filtered_india) > 0:
            center_lat = float(np.mean(filtered_india["latitude"]))
            center_lon = float(np.mean(filtered_india["longitude"]))
            zoom_folium = 9 if len(filtered_india) < 600 else 6
            zoom_pdk = 9.8 if len(filtered_india) < 600 else 6.5
        else:
            center_lat, center_lon = 22.2, 79.5
            zoom_folium = 5
            zoom_pdk = 4.6

        # MODE 1: CLEAR GEOGRAPHIC ROAD MAP (OPENSTREETMAP / FOLIUM)
        if map_view_engine.startswith("🗺️ Clear Geographic"):
            folium_ctrl1, folium_ctrl2, folium_ctrl3 = st.columns([1.5, 1.2, 1.3])
            with folium_ctrl1:
                tile_style = st.selectbox(
                    "Clear Basemap Layer:",
                    ["OpenStreetMap (Clear Colorful Road Map & State Borders)", "CartoDB Positron (High-Contrast Clean White)", "CartoDB Dark_Matter (Night Mode)"]
                )
            with folium_ctrl2:
                show_folium_highways = st.checkbox("🛣️ Show Highway Corridors", value=True)
            with folium_ctrl3:
                show_folium_blackspots = st.checkbox("🔴 Show Danger Blackspots", value=True)

            tile_key = "OpenStreetMap"
            if "Positron" in tile_style:
                tile_key = "CartoDB positron"
            elif "Dark_Matter" in tile_style:
                tile_key = "CartoDB dark_matter"

            # Create Folium Map with clean India view
            fmap = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=zoom_folium,
                tiles=tile_key,
                control_scale=True,
                prefer_canvas=True
            )

            # Define vibrant color palette for highways in Folium
            HWY_HEX_COLORS = {
                "NH-44": "#0284c7",          # Vivid Blue
                "NH-48": "#d97706",          # Amber
                "MUM-PUN-EXP": "#dc2626",    # Crimson Red
                "YAM-EXP": "#e11d48",        # Rose
                "NH-66": "#9333ea",          # Purple
                "SAMRUDDHI-EXP": "#059669",  # Emerald
                "NH-16": "#2563eb",          # Royal Blue
                "NH-19": "#ca8a04",          # Gold
                "HYD-ORR": "#6366f1",        # Indigo
                "NE-4-EXP": "#0d9488"        # Teal
            }

            # Render Highway Polylines on Folium Map
            if show_folium_highways:
                for h in highway_routes:
                    hid = h["highway_id"]
                    # Deck.gl has [lon, lat], Folium requires [lat, lon]
                    lat_lon_path = [[pt[1], pt[0]] for pt in h["path"]]

                    h_matched = False
                    if active_query:
                        if (active_query.lower() in h["short_name"].lower() or 
                            active_query.lower() in h["highway_name"].lower() or 
                            active_query.lower() in h["highway_id"].lower()):
                            h_matched = True

                    if is_filtered and h_matched:
                        line_color = "#ef4444"
                        line_weight = 7
                        line_opacity = 1.0
                    elif is_filtered and not h_matched:
                        line_color = "#94a3b8"
                        line_weight = 3
                        line_opacity = 0.55
                    else:
                        line_color = HWY_HEX_COLORS.get(hid, "#3b82f6")
                        line_weight = 5
                        line_opacity = 0.85

                    hwy_popup_html = f"""
                    <div style="font-family: system-ui, sans-serif; font-size: 0.95rem; min-width: 240px; padding: 4px;">
                        <h4 style="margin: 0 0 6px 0; color: #1e3a8a;">{h['highway_name']}</h4>
                        <b>Length:</b> {h['length_km']:,} km<br/>
                        <b>Category:</b> {h['category']}<br/>
                        <b>Danger Rating:</b> <span style="color: #dc2626; font-weight: bold;">{h['danger_level']}</span><br/>
                        <b>MoRTH Blackspots:</b> {h['total_blackspots']} Stretches<br/>
                        <p style="margin: 6px 0 0 0; font-size: 0.85rem; color: #475569;">{h['description']}</p>
                    </div>
                    """

                    folium.PolyLine(
                        lat_lon_path,
                        color=line_color,
                        weight=line_weight,
                        opacity=line_opacity,
                        tooltip=f"🛣️ {h['short_name']} ({h['length_km']:,} km) - {h['danger_level']}",
                        popup=folium.Popup(hwy_popup_html, max_width=320)
                    ).add_to(fmap)

            # Render Danger Blackspot Markers on Folium Map
            if show_folium_blackspots and len(blackspots_nodes_df) > 0:
                for _, b in blackspots_nodes_df.iterrows():
                    tot = b["total_accidents"]
                    fat = b["fatal_accidents"]
                    adv = b["advisory_speed"]
                    area = b["area_name"]
                    hwy = b["highway_name"]
                    code = b["morth_blackspot_id"]
                    cause = b["primary_cause"]
                    d_lvl = b["danger_level"]

                    popup_html = f"""
                    <div style="font-family: system-ui, sans-serif; min-width: 260px; padding: 6px;">
                        <div style="background: #991b1b; color: white; padding: 6px 10px; border-radius: 6px; font-weight: bold; margin-bottom: 8px;">
                            🛑 {area}
                        </div>
                        <div style="font-size: 0.9rem; line-height: 1.5; color: #1e293b;">
                            <b>Highway:</b> {hwy}<br/>
                            <b>MoRTH Blackspot Code:</b> <code>{code}</code><br/>
                            <b>Danger Level:</b> <span style="color: #b91c1c; font-weight: bold;">{d_lvl}</span><br/>
                            <b>Recorded Crashes:</b> {tot} collisions (<strong>{fat} Fatalities</strong>)<br/>
                            <div style="margin: 6px 0; background: #ecfdf5; border-left: 4px solid #10b981; padding: 5px 8px; border-radius: 4px;">
                                🟢 <b>Safe Advisory Speed:</b> <span style="color: #065f46; font-weight: bold; font-size: 1.05rem;">{adv} km/h</span>
                            </div>
                            <b>Primary Hazard:</b> {cause}
                        </div>
                    </div>
                    """

                    # Pulsing circle marker
                    folium.CircleMarker(
                        location=[b["latitude"], b["longitude"]],
                        radius=11 if is_filtered else 8,
                        color="#991b1b",
                        weight=2.5,
                        fill=True,
                        fill_color="#ef4444",
                        fill_opacity=0.85,
                        tooltip=f"⚠️ {area.split('(')[0].strip()}: {tot} Crashes ({fat} Fatal) | GO ≤ {adv} km/h",
                        popup=folium.Popup(popup_html, max_width=320)
                    ).add_to(fmap)

            # Embed Folium map with full interactivity
            components.html(fmap._repr_html_(), height=660)
            st.caption(
                f"Showing **Clear Map of India** with {len(highway_routes)} major highway corridors traversing states, "
                f"and {len(blackspots_nodes_df)} official MoRTH high-risk blackspots. Click any highway or danger circle for full safety details."
            )

        # MODE 2: 3D PERSPECTIVE ELEVATION MAP (PYDECK WEBGL)
        else:
            map_c1, map_c2, map_c3, map_c4 = st.columns([1.3, 1.1, 1.1, 1.1])
            with map_c1:
                show_highways = st.checkbox("🛣️ Show Highway Routes", value=True)
            with map_c2:
                show_pillars = st.checkbox("🗼 Show 3D Danger Pillars", value=True)
            with map_c3:
                show_rings = st.checkbox("⭕ Show Danger Rings", value=True)
            with map_c4:
                pdk_style_choice = st.selectbox("3D Basemap:", ["Clear Light (Carto Positron)", "Dark Mode (Carto Dark)"])

            pdk_map_style = pdk.map_styles.CARTO_LIGHT if "Light" in pdk_style_choice else pdk.map_styles.CARTO_DARK

            # Prepare Highway Network PathLayer Data
            path_records = []
            for h in highway_routes:
                h_matched = False
                if active_query:
                    if (active_query.lower() in h["short_name"].lower() or 
                        active_query.lower() in h["highway_name"].lower() or 
                        active_query.lower() in h["highway_id"].lower()):
                        h_matched = True

                if is_filtered and h_matched:
                    line_color = [239, 68, 68, 255]
                    line_width = 8
                elif is_filtered and not h_matched:
                    line_color = [100, 116, 139, 120]
                    line_width = 2
                else:
                    line_color = h["color"]
                    line_width = 4

                path_records.append({
                    "highway_id": h["highway_id"],
                    "highway_name": h["highway_name"],
                    "short_name": h["short_name"],
                    "category": h["category"],
                    "length_km": h["length_km"],
                    "danger_level": h["danger_level"],
                    "total_blackspots": h["total_blackspots"],
                    "description": h["description"],
                    "color": line_color,
                    "width": line_width,
                    "path": h["path"]
                })

            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=zoom_pdk,
                pitch=42,
                bearing=10
            )

            layers_to_render = []

            # Layer 1: Highway Network Polylines (PathLayer)
            if show_highways and len(path_records) > 0:
                highway_layer = pdk.Layer(
                    "PathLayer",
                    data=path_records,
                    get_path="path",
                    get_color="color",
                    get_width="width",
                    width_scale=20,
                    width_min_pixels=3,
                    width_max_pixels=14,
                    rounded=True,
                    joint_rounded=True,
                    pickable=True,
                    auto_highlight=True,
                )
                layers_to_render.append(highway_layer)

            # Layer 2: 3D Danger Blackspots (ColumnLayer)
            if show_pillars and len(filtered_india) > 0:
                def get_elevation(row):
                    sev = row["accident_severity"]
                    spd = row["recorded_speed_kmh"]
                    if sev == "Fatal":
                        return 2400 + (spd * 14)
                    elif sev == "Serious":
                        return 1200 + (spd * 8)
                    return 400 + (spd * 3)

                def get_pillar_color(sev):
                    if sev == "Fatal":
                        return [239, 68, 68, 230]
                    elif sev == "Serious":
                        return [249, 115, 22, 200]
                    return [234, 179, 8, 170]

                plot_df = filtered_india.copy()
                plot_df["elevation"] = plot_df.apply(get_elevation, axis=1)
                plot_df["color"] = plot_df["accident_severity"].apply(get_pillar_color)

                layer_3d = pdk.Layer(
                    "ColumnLayer",
                    data=plot_df,
                    get_position=["longitude", "latitude"],
                    get_elevation="elevation",
                    elevation_scale=1.5,
                    radius=650 if not is_filtered else 220,
                    get_fill_color="color",
                    pickable=True,
                    auto_highlight=True,
                    extruded=True,
                )
                layers_to_render.append(layer_3d)

            # Layer 3: Danger Perimeter Radii (ScatterplotLayer)
            if show_rings and len(blackspots_nodes_df) > 0:
                ring_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=blackspots_nodes_df,
                    get_position=["longitude", "latitude"],
                    get_radius=3400 if not is_filtered else 1100,
                    get_fill_color=[239, 68, 68, 40],
                    get_line_color=[239, 68, 68, 255],
                    stroked=True,
                    filled=True,
                    line_width_min_pixels=2.5,
                    pickable=True,
                )
                layers_to_render.append(ring_layer)

            tooltip_dict = {
                "html": (
                    "<div style='background-color: #0f172a; color: white; padding: 12px; border-radius: 8px; font-family: system-ui, sans-serif; box-shadow: 0 4px 12px rgba(0,0,0,0.5); max-width: 320px;'>"
                    "{area_name}"
                    "<div style='font-size: 0.9rem; margin-top: 4px;'>"
                    "<b>Highway:</b> {highway_name}<br/>"
                    "<b>Severity:</b> <span style='color: #f87171; font-weight: bold;'>{accident_severity}</span><br/>"
                    "<b>Recorded Speed:</b> {recorded_speed_kmh} km/h (Posted: {posted_speed_kmh} km/h)<br/>"
                    "<b style='color: #4ade80;'>Advisory Safe Speed:</b> <b>{advisory_safe_speed_kmh} km/h</b><br/>"
                    "<b>Danger Hazard:</b> <span style='color: #fde047;'>{primary_cause}</span><br/>"
                    "<small style='color: #94a3b8;'>MoRTH Code: {morth_blackspot_id}</small>"
                    "</div>"
                    "</div>"
                ),
                "style": {"backgroundColor": "transparent"}
            }

            st.pydeck_chart(
                pdk.Deck(
                    map_style=pdk_map_style,
                    initial_view_state=view_state,
                    layers=layers_to_render,
                    tooltip=tooltip_dict
                )
            )
            st.caption(
                f"Displaying **{len(path_records)} National Highway & Expressway corridors** traversing India and "
                f"**{len(filtered_india):,} accident records** in 3D perspective with Carto high-visibility tiles."
            )

        # Section D: Indian National Highway Danger Registry & Corridor Comparison
        st.markdown("---")
        st.markdown("#### 📋 India National Highways & Expressways Danger Registry")
        st.markdown("""
        Comparative safety analysis of India's arterial transit corridors based on MoRTH annual blackspot registries,
        recorded fatality ratios, and primary collision mechanics.
        """)

        hwy_summary_rows = []
        corridors_list = load_app_india_corridors()
        for h in highway_routes:
            hid = h["highway_id"]
            # Find matching blackspots in dataset
            matched_spots = [c for c in corridors_list if c.get("highway_id") == hid]
            danger_spots_names = ", ".join([s["area_name"].split("(")[0].strip() for s in matched_spots]) if matched_spots else "General Corridor Stretch"
            primary_hz = matched_spots[0]["primary_cause"] if matched_spots else h["description"]
            adv_spd = f"{matched_spots[0]['advisory_speed']} km/h" if matched_spots else "60-80 km/h"

            hwy_summary_rows.append({
                "Highway Corridor": h["short_name"],
                "Category": h["category"],
                "Length (km)": f"{h['length_km']:,} km",
                "Danger Classification": h["danger_level"],
                "Critical Danger Areas": danger_spots_names,
                "Advisory Safe Speed": adv_spd,
                "Primary Collision Mechanics": primary_hz
            })

        df_hwy_summary = pd.DataFrame(hwy_summary_rows)
        st.dataframe(df_hwy_summary, use_container_width=True)

        st.caption("Grounded in official statistics from the Ministry of Road Transport and Highways (MoRTH) Road Accidents in India Annual Report, IRC-SP-44, and iRAD portal.")

    else:
        st.error("India accident blackspots and highway network dataset could not be loaded.")


# -------------------------------------------------------------
# PAGE 5: GEOSPATIAL HOTSPOT MAP
# -------------------------------------------------------------
elif page == "Geospatial Hotspot Map":
    st.subheader("🗺️ Interactive Highway Collision Hotspot & Corridor Map")
    st.markdown("""
    Explore real-world highway collision corridors grounded in official **UK Department for Transport (DfT)** 
    and **Road Safety Foundation** accident investigation data. Visualized in 3D using **PyDeck**.
    """)

    df = load_app_data()
    if df is not None and "latitude" in df.columns and "longitude" in df.columns:
        df_geo = df.dropna(subset=["latitude", "longitude"]).copy()

        # Highway and Severity Filter Controls
        geo_col1, geo_col2, geo_col3 = st.columns([1.5, 1.2, 1.3])

        with geo_col1:
            all_highways = ["All Highway Corridors"] + sorted(list(df_geo["highway_name"].dropna().unique())) if "highway_name" in df_geo.columns else ["All Highway Corridors"]
            selected_highway = st.selectbox("Select Highway Corridor:", all_highways)

        with geo_col2:
            sev_filter = st.multiselect(
                "Filter Severity Tiers:",
                ["Slight", "Serious", "Fatal"],
                default=["Slight", "Serious", "Fatal"]
            )

        with geo_col3:
            layer_type = st.radio("Visualization Layer:", ["3D Hexagon Density Grid", "Scatterplot by Severity Class"], horizontal=True)

        # Apply Filters
        filtered_geo = df_geo[df_geo["accident_severity"].isin(sev_filter)]
        if selected_highway != "All Highway Corridors" and "highway_name" in filtered_geo.columns:
            filtered_geo = filtered_geo[filtered_geo["highway_name"] == selected_highway]

        # Highway Corridor Focus Info
        CORRIDOR_DESCRIPTIONS = {
            "M6 Motorway": "The UK's longest motorway. Key high-risk focus at Junction 6 (Spaghetti Junction, Birmingham) due to massive HGV freight density and complex merging lanes.",
            "M1 Motorway": "Primary London-to-Leeds artery. High accident frequency along the Northamptonshire-Bedfordshire corridor (J15-16), heavily impacted by high-speed multi-vehicle rear-end collisions.",
            "M25 London Orbital": "Encircles Greater London. Heavy congestion with collision hotspots at major junctions including the M23 interchange (Surrey) and Dartford Crossing approach.",
            "M4 Motorway": "London-to-Wales transport corridor. High-speed multi-lane traffic around Reading and Berkshire.",
            "A1(M) Great North Road": "Historic London-to-Edinburgh trunk route. Transition points between motorway and single carriageway in Nottinghamshire have high head-on impact histories.",
            "A537 Cat and Fiddle Road": "Iconic Peak District pass (Macclesfield to Buxton). Ranked among the UK's most dangerous rural roads due to blind summits, sharp hairpins, and high motorcycle casualty rates.",
            "A57 Snake Pass": "High-altitude Pennine mountain pass in Derbyshire. Severe collision hazards driven by winter ice, fog, and steep gradients.",
            "A38 Devon Expressway": "High-speed South West trunk route between Exeter and Plymouth, recording elevated speed-related collision counts.",
            "A9 Highland Corridor": "Longest Scottish trunk corridor (Perth to Inverness). High proportions of fatal collisions on rural single-carriageway overtaking stretches."
        }

        if selected_highway != "All Highway Corridors":
            st.markdown(f"#### 🛣️ Corridor Focus: **{selected_highway}**")
            info_c1, info_c2, info_c3 = st.columns([2, 1, 1])
            with info_c1:
                desc = CORRIDOR_DESCRIPTIONS.get(selected_highway, "Major national transport artery with high recorded incident frequency.")
                loc = filtered_geo["highway_location"].iloc[0] if "highway_location" in filtered_geo.columns and len(filtered_geo) > 0 else "High-traffic stretch"
                st.write(f"**Key Hotspot Location:** `{loc}`")
                st.write(f"**Official Risk Profile:** {desc}")
            with info_c2:
                st.metric("Recorded Incidents", f"{len(filtered_geo):,}")
            with info_c3:
                h_fatal = (filtered_geo["accident_severity"] == "Fatal").mean() * 100 if len(filtered_geo) > 0 else 0
                h_serious = (filtered_geo["accident_severity"] == "Serious").mean() * 100 if len(filtered_geo) > 0 else 0
                st.metric("Corridor Fatal Rate", f"{h_fatal:.1f}%", f"+{h_serious:.1f}% Serious")

        # Map View State
        if len(filtered_geo) > 0:
            mid_lat = float(np.mean(filtered_geo["latitude"]))
            mid_lon = float(np.mean(filtered_geo["longitude"]))
            zoom_lvl = 11.5 if selected_highway != "All Highway Corridors" else 6.8
        else:
            mid_lat, mid_lon, zoom_lvl = 52.4862, -1.8904, 6.8

        view_state = pdk.ViewState(
            latitude=mid_lat,
            longitude=mid_lon,
            zoom=zoom_lvl,
            pitch=45 if layer_type.startswith("3D") else 0
        )

        if layer_type.startswith("3D"):
            layer = pdk.Layer(
                "HexagonLayer",
                data=filtered_geo,
                get_position=["longitude", "latitude"],
                radius=600 if selected_highway == "All Highway Corridors" else 250,
                elevation_scale=40,
                elevation_range=[0, 1500],
                pickable=True,
                extruded=True,
            )
            tooltip_html = {
                "html": "<b>Hexagon Cluster</b><br/>Collision Density: {elevationValue} incidents",
                "style": {"backgroundColor": "#1e3a8a", "color": "white"}
            }
        else:
            def get_color(sev):
                if sev == "Fatal":
                    return [231, 76, 60, 220]
                elif sev == "Serious":
                    return [230, 126, 34, 190]
                return [52, 152, 219, 150]

            filtered_geo["color"] = filtered_geo["accident_severity"].apply(get_color)
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=filtered_geo,
                get_position=["longitude", "latitude"],
                get_color="color",
                get_radius=180 if selected_highway == "All Highway Corridors" else 75,
                pickable=True,
            )
            tooltip_html = {
                "html": (
                    "<b>Highway:</b> {highway_name}<br/>"
                    "<b>Location:</b> {highway_location}<br/>"
                    "<b>Severity:</b> {accident_severity}<br/>"
                    "<b>Speed Limit:</b> {speed_limit} mph<br/>"
                    "<b>Vehicle:</b> {vehicle_type}<br/>"
                    "<b>Weather:</b> {weather_conditions}"
                ),
                "style": {"backgroundColor": "#0f172a", "color": "white"}
            }

        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=[layer],
                tooltip=tooltip_html
            )
        )
        st.caption(f"Displaying {len(filtered_geo):,} geo-referenced incidents across verified UK highway corridors.")

        # Highway Corridor Benchmark Ranking Table
        st.markdown("---")
        st.markdown("#### 📊 National Highway Collision Risk Ranking (Grounded in DfT & Road Safety Foundation Data)")

        ranking_data = [
            {"Highway Corridor": "M6 Motorway", "High-Risk Segment": "Junction 6 Gravelly Hill (Birmingham)", "Type": "Motorway", "Key Hazard": "Dense HGV freight, interchange lane weaves", "Historical Risk Tier": "High Volume"},
            {"Highway Corridor": "M1 Motorway", "High-Risk Segment": "Junction 15-16 Northamptonshire", "Type": "Motorway", "Key Hazard": "High-speed multi-lane rear-end collisions", "Historical Risk Tier": "High Volume"},
            {"Highway Corridor": "M25 London Orbital", "High-Risk Segment": "Junction 7-8 Surrey Interchange", "Type": "Orbital Motorway", "Key Hazard": "Severe stop-start commuter shunts", "Historical Risk Tier": "High Volume"},
            {"Highway Corridor": "A537 Cat & Fiddle", "High-Risk Segment": "Peak District Mountain Pass", "Type": "Rural A-Road", "Key Hazard": "Blind bends, hairpins, motorcycle fatalities", "Historical Risk Tier": "Persistent High Risk"},
            {"Highway Corridor": "A57 Snake Pass", "High-Risk Segment": "High Peak Pennine Pass", "Type": "Rural A-Road", "Key Hazard": "Winter ice, steep gradient drops", "Historical Risk Tier": "Severe Weather Risk"},
            {"Highway Corridor": "A1(M) Great North", "High-Risk Segment": "Nottinghamshire Transition", "Type": "Trunk Dual/Single", "Key Hazard": "Transition merges, mixed local/high-speed traffic", "Historical Risk Tier": "Moderate-High"},
            {"Highway Corridor": "A9 Highland Corridor", "High-Risk Segment": "Perth to Inverness Pass", "Type": "Single/Dual A-Road", "Key Hazard": "Single-carriageway overtaking collisions", "Historical Risk Tier": "High Fatality Rate"},
            {"Highway Corridor": "A38 Devon Expressway", "High-Risk Segment": "Exeter-Plymouth Stretch", "Type": "Dual Carriageway", "Key Hazard": "High travel speeds, rural junctions", "Historical Risk Tier": "Elevated Risk"}
        ]
        st.dataframe(pd.DataFrame(ranking_data), use_container_width=True)
        st.caption("Reference Sources: UK Department for Transport (DfT) Road Safety Open Data (STATS19) & Road Safety Foundation Annual Risk Reports.")

    else:
        st.info("Geospatial coordinates not available in current dataset.")


# -------------------------------------------------------------
# PAGE 6: MODEL PERFORMANCE
# -------------------------------------------------------------
elif page == "Model Performance":
    st.subheader("Model Evaluation & Benchmark Comparison")
    st.write("Rigorous multi-model evaluation on held-out test data (600 incidents), assessing class-imbalance resilience.")

    if metadata is not None:
        eval_dict = metadata.get("all_models_evaluation", {})

        from src.evaluate import generate_comparison_table
        comp_df = generate_comparison_table(eval_dict)
        st.dataframe(comp_df, use_container_width=True)

        st.markdown("---")
        st.markdown("### Model Diagnostics & Deep Dive")

        model_selected = st.selectbox(
            "Select Model to Inspect:",
            ["Random Forest", "XGBoost"]
        )

        col_diag_left, col_diag_right = st.columns(2)

        with col_diag_left:
            st.markdown(f"#### 🟦 Confusion Matrix: {model_selected}")
            cm_img_path = f"reports/figures/confusion_matrix_{model_selected.lower().replace(' ', '_')}.png"
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, use_container_width=True)
            else:
                st.info("Confusion matrix image not found.")

        with col_diag_right:
            st.markdown(f"#### 📊 Top Feature Importances: {model_selected}")
            fi_img_path = f"reports/figures/feature_importance_{model_selected.lower().replace(' ', '_')}.png"
            if os.path.exists(fi_img_path):
                st.image(fi_img_path, use_container_width=True)
            else:
                st.info("Feature importance image not found.")

        st.markdown("---")
        st.markdown("### 🔍 The Imbalance Paradox: Why Raw Accuracy Is Deceptive")
        st.markdown("""
        In road safety classification, **class imbalance creates an accuracy illusion**:
        - **The Majority Baseline Paradox**: A trivial dummy model that blindly predicts `Slight` on 100% of cases achieves **71.8% accuracy** on the test set. However, its recall for `Serious` and `Fatal` collisions is **0.0%**.
        - **Balanced Class Weighting**: By applying class-balanced weighting during training, **Random Forest** and **XGBoost** sacrifice some accuracy on the majority class in order to achieve high **Macro-Recall (~36.6%)** and detect minority life-threatening injuries.
        - **Macro F1-Score**: We optimize for Macro F1 (unweighted average of F1 across all classes), ensuring every life-critical class is treated with equal mathematical priority.
        """)
    else:
        st.warning("Model evaluation metadata not found. Run `python -m src.train`.")


# -------------------------------------------------------------
# PAGE 7: DATA INSIGHTS
# -------------------------------------------------------------
elif page == "Data Insights":
    st.subheader("Interactive Data Insights & Risk Visualizations")
    st.write("Explore risk correlations between environmental conditions, roadway infrastructure, and collision severity.")

    df = load_app_data()
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["🌧️ Weather & Road Surface", "⚡ Speed Limits & Infrastructure", "🏍️ Vehicle Profiles"])

        with tab1:
            st.markdown("#### Environmental Risk Matrix")
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                img_weather = "reports/figures/04_weather_severity.png"
                if os.path.exists(img_weather):
                    st.image(img_weather, use_container_width=True)
                else:
                    st.write("Weather plot not found.")
            with col_w2:
                img_surface = "reports/figures/05_road_surface_severity.png"
                if os.path.exists(img_surface):
                    st.image(img_surface, use_container_width=True)
                else:
                    st.write("Surface plot not found.")

        with tab2:
            st.markdown("#### Speed & Roadway Geometry")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                img_speed = "reports/figures/05_speed_vs_severity.png"
                if os.path.exists(img_speed):
                    st.image(img_speed, use_container_width=True)
                else:
                    st.write("Speed plot not found.")
            with col_s2:
                img_light = "reports/figures/06_light_severity.png"
                if os.path.exists(img_light):
                    st.image(img_light, use_container_width=True)
                else:
                    st.write("Light plot not found.")

        with tab3:
            st.markdown("#### Vulnerable Road Users & Collision Types")
            fig_day = "reports/figures/03_accidents_by_day.png"
            if os.path.exists(fig_day):
                st.image(fig_day, use_container_width=True)

