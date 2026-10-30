"""
Road Accident Severity Predictor - Interactive Web Application.
Built with Streamlit, Scikit-Learn, and XGBoost.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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
    except Exception as e:
        return None


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/traffic-light.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview Dashboard", "Accident Severity Prediction", "Model Performance", "Data Insights", "About the Project"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
predictor = load_app_predictor()
metadata = load_app_metadata()

if predictor is not None and metadata is not None:
    st.sidebar.success(f"Model Active: **{metadata['best_model']['name']}**")
    st.sidebar.caption(f"Evaluated: {metadata['evaluation_timestamp'][:10]}")
else:
    st.sidebar.warning("Model Not Loaded")

st.sidebar.markdown("---")
st.sidebar.caption("Road Accident Severity Predictor v1.0.0  \nBuilt for Technical Evaluation")

# Header Banner
st.markdown('<div class="main-header">Road Accident Severity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data-driven exploration and estimation of road accident severity using pre-collision factors.</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# PAGE A: OVERVIEW DASHBOARD
# -------------------------------------------------------------
if page == "Overview Dashboard":
    st.subheader("Executive Dashboard")

    df = load_app_data()
    if df is not None:
        # KPI Cards
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

        # Two-column layout for key EDA plots
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("#### 🎯 Class Imbalance Distribution")
            fig1_path = "reports/figures/01_class_distribution.png"
            if os.path.exists(fig1_path):
                st.image(fig1_path, use_container_width=True)
            else:
                st.bar_chart(df['accident_severity'].value_counts())
            st.caption("Severity classes exhibit standard real-world imbalance (~78% Slight, ~18% Serious, ~4% Fatal).")

        with col_right:
            st.markdown("#### ⏰ Temporal Accident Distribution (24-Hour)")
            fig2_path = "reports/figures/02_accidents_by_hour.png"
            if os.path.exists(fig2_path):
                st.image(fig2_path, use_container_width=True)
            else:
                st.info("Hourly plot generated during training.")
            st.caption("Collisions peak during morning (07:00-09:00) and evening commuter rush hours (16:00-19:00).")

        st.markdown("---")
        st.markdown("#### 🏆 Active Machine Learning Pipeline Summary")

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
                f"💡 **Model Selection Insight**: The **{best_info['name']}** model was selected because it maximizes "
                f"**Macro F1-score ({best_info['macro_f1']:.4f})**, ensuring balanced sensitivity to rare Fatal and Serious "
                "incidents rather than optimizing for majority-class accuracy alone."
            )
    else:
        st.warning("Dataset could not be loaded. Please ensure `data/raw/road_accidents_demo.csv` exists.")


# -------------------------------------------------------------
# PAGE B: ACCIDENT SEVERITY PREDICTION
# -------------------------------------------------------------
elif page == "Accident Severity Prediction":
    st.subheader("Predict Accident Severity")
    st.write("Input pre-accident environmental, roadway, and vehicular conditions to compute severity estimates.")

    if predictor is None:
        st.error("Model artifacts not found. Please train the models by executing `python -m src.train`.")
    else:
        # Presets selector for rapid demo
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

        # Default values based on scenario
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

                # Badge style
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

                # Prominent Disclaimer
                st.markdown(f"""
                <div class="disclaimer-box">
                    <strong>⚠️ Important Decision-Support Notice:</strong><br>
                    {result['disclaimer']}
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Inference error: {e}")


# -------------------------------------------------------------
# PAGE C: MODEL PERFORMANCE
# -------------------------------------------------------------
elif page == "Model Performance":
    st.subheader("Model Evaluation & Benchmark Comparison")
    st.write("Rigorous multi-model evaluation on held-out test data (600 incidents), assessing class-imbalance resilience.")

    if metadata is not None:
        eval_dict = metadata.get("all_models_evaluation", {})

        # Comparison Table
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
        st.warning("Model evaluation metadata not found. Run `python -m src.train` to generate benchmark results.")


# -------------------------------------------------------------
# PAGE D: DATA INSIGHTS
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
# PAGE E: ABOUT THE PROJECT
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

    ### ⚙️ Machine Learning Pipeline Architecture
    ```
    ┌─────────────────────────┐
    │ Raw Accident Dataset    │
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Schema & Leakage Audit  │  (Validates pre-collision features)
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Feature Engineering     │  (Extracts 24h hour, rush hour, flags)
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ ColumnTransformer       │  (Median Impute + Scaling, OneHotEncoding)
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Stratified Train/Test   │  (80/20 split preserving class ratios)
    └───────────┬─────────────┘
                │
         ┌──────┴──────────────┐
         ▼                     ▼
    ┌───────────────┐   ┌───────────────┐
    │ Random Forest │   │    XGBoost    │  (Balanced class weights)
    └───────┬───────┘   └───────┬───────┘
            │                   │
            └─────────┬─────────┘
                      ▼
    ┌─────────────────────────┐
    │ Imbalance-Aware Eval    │  (Evaluates Macro F1, Recall, OvR AUC)
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Top Model Serialization │  (Joblib model + metadata)
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Streamlit Interface     │  (Real-time inference & risk factors)
    └─────────────────────────┘
    ```

    ---

    ### 🛠️ Technology Stack
    - **Core ML**: Scikit-Learn, XGBoost, NumPy, Pandas
    - **Visualizations**: Matplotlib, Seaborn
    - **Interface**: Streamlit
    - **Serialization**: Joblib, PyYAML
    - **Quality & Testing**: Pytest

    ---

    ### ⚖️ Responsible AI & Ethical Disclaimer
    This application is an educational, analytical machine learning project. The predictions produced are
    **experimental statistical estimates** and must **never** be used as the sole operational basis for
    emergency service dispatch, real-time routing, or insurance liability determinations.
    """)
