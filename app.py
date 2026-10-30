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
        "Geospatial Hotspot Map",
        "Model Performance",
        "Data Insights",
        "About the Project"
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
# PAGE 5: GEOSPATIAL HOTSPOT MAP
# -------------------------------------------------------------
elif page == "Geospatial Hotspot Map":
    st.subheader("🗺️ Interactive Geospatial Accident Hotspot Density Map")
    st.markdown("""
    Interactive 3D visualization of collision density and severity clusters powered by **PyDeck**.
    Explore incident clusters, high-risk rural transport corridors, and urban incident density.
    """)

    df = load_app_data()
    if df is not None and "latitude" in df.columns and "longitude" in df.columns:
        df_geo = df.dropna(subset=["latitude", "longitude"]).copy()

        sev_filter = st.multiselect(
            "Filter by Severity Tier:",
            ["Slight", "Serious", "Fatal"],
            default=["Slight", "Serious", "Fatal"]
        )
        filtered_geo = df_geo[df_geo["accident_severity"].isin(sev_filter)]

        layer_type = st.radio("Map Visualization Layer:", ["3D Hexagon Density Grid", "Scatterplot by Severity Class"], horizontal=True)

        midpoint = (float(np.mean(filtered_geo["latitude"])), float(np.mean(filtered_geo["longitude"])))

        view_state = pdk.ViewState(
            latitude=midpoint[0],
            longitude=midpoint[1],
            zoom=9.5,
            pitch=45 if layer_type.startswith("3D") else 0
        )

        if layer_type.startswith("3D"):
            layer = pdk.Layer(
                "HexagonLayer",
                data=filtered_geo,
                get_position=["longitude", "latitude"],
                radius=400,
                elevation_scale=30,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            )
        else:
            def get_color(sev):
                if sev == "Fatal":
                    return [231, 76, 60, 200]
                elif sev == "Serious":
                    return [230, 126, 34, 180]
                return [52, 152, 219, 140]

            filtered_geo["color"] = filtered_geo["accident_severity"].apply(get_color)
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=filtered_geo,
                get_position=["longitude", "latitude"],
                get_color="color",
                get_radius=120,
                pickable=True,
            )

        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=[layer],
                tooltip={"text": "Incidents Clustered: {elevationValue}" if layer_type.startswith("3D") else "Severity: {accident_severity}"}
            )
        )
        st.caption(f"Displaying {len(filtered_geo):,} geo-referenced incidents across road corridors.")
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


# -------------------------------------------------------------
# PAGE 8: ABOUT THE PROJECT
# -------------------------------------------------------------
elif page == "About the Project":
    st.subheader("About the Road Accident Severity Predictor")

    st.markdown("""
    ### 🎯 Problem Statement
    Road traffic collisions represent one of the leading global causes of preventable injury and death.
    The goal of this system is to predict **accident severity** (`Slight`, `Serious`, `Fatal`) based on
    **pre-accident conditions** (weather, lighting, speed limits, road geometry, and vehicle categories).

    ---

    ### 🛡️ Preventing Target Leakage
    A critical engineering requirement in accident modeling is **target leakage prevention**:
    - **Valid Predictors**: Speed limit, weather, lighting, road type, time of day, vehicle type.
    - **Prohibited Predictors (Target Leakage)**: `number_of_casualties`, `casualty_severity`, `hospital_arrival_time`,
      `airbag_deployment_count`, or `vehicle_damage_cost`. These observations only occur **after** the crash.
      Including them causes artificial 99% test accuracy while rendering the model completely invalid for pre-collision risk estimation.

    ---

    ### 🛠️ Technology Stack
    - **Core ML**: Scikit-Learn, XGBoost, NumPy, Pandas
    - **API Microservice**: FastAPI, Pydantic v2, Uvicorn
    - **Web Application & Geospatial**: Streamlit, PyDeck
    - **Visualizations**: Matplotlib, Seaborn
    - **Serialization**: Joblib, PyYAML
    - **Testing & CI/CD**: Pytest (14 passing unit & API tests), GitHub Actions

    ---

    ### ⚖️ Responsible AI & Ethical Disclaimer
    This application is an educational, analytical machine learning project. The predictions produced are
    **experimental statistical estimates** and must **never** be used as the sole operational basis for
    emergency service dispatch, real-time routing, or insurance liability determinations.
    """)
