# Dataset Documentation: Road Accident Severity Predictor

## 1. Overview & Data Philosophy

Predicting road accident severity requires strict attention to **data integrity and leakage prevention**:
- **Objective**: Predict accident severity (`Slight`, `Serious`, `Fatal`) using conditions present **before or at the onset of an accident**.
- **Leakage Prevention**: Variables that only become available **after** the collision (such as `number_of_casualties`, `casualty_severity`, `hospital_arrival_time`, `vehicle_towing_status`, `airbag_deployment_count`, or `claim_amount`) are strictly excluded from the feature set. Utilizing post-accident information would cause severe data leakage and artificially inflate test metrics while making the model useless for pre-collision risk assessment.

---

## 2. Included Benchmark Demonstration Dataset (`data/raw/road_accidents_demo.csv`)

To make this repository fully self-contained, reproducible, and ready for immediate testing, training, and deployment out of the box, we provide a structured benchmark demonstration dataset:
- **Filename**: `data/raw/road_accidents_demo.csv`
- **Total Records**: 3,000 incidents
- **Data Source Note**: Clearly labeled as a **calibrated benchmark demonstration dataset** modeled on real-world accident distribution metrics published in the UK Department for Transport (DfT) STATS19 open records and international road safety research.
- **Class Imbalance Realism**:
  - `Slight`: ~78% - 82% of incidents (minor injuries, scrapes, outpatient checks)
  - `Serious`: ~14% - 18% of incidents (fractures, severe lacerations, hospital admissions)
  - `Fatal`: ~2% - 4% of incidents (fatalities within 30 days)

### Feature Dictionary

| Feature Name | Type | Description | Values / Range |
| :--- | :--- | :--- | :--- |
| `accident_id` | String | Unique identifier for the incident | e.g. `ACC-2026-0001` |
| `accident_time` | String | Timestamp of accident (HH:MM) | `00:00` to `23:59` |
| `day_of_week` | Categorical | Day on which the collision occurred | `Monday`, `Tuesday`, ..., `Sunday` |
| `speed_limit` | Numerical | Posted speed limit in mph | `20`, `30`, `40`, `50`, `60`, `70` |
| `weather_conditions` | Categorical | Atmospheric weather at time of incident | `Fine no high winds`, `Raining no high winds`, `Raining + high winds`, `Snowing`, `Fog or mist`, `Other` |
| `road_surface_conditions`| Categorical | Condition of the road surface | `Dry`, `Wet or damp`, `Snow`, `Frost or ice`, `Flood over 3cm` |
| `light_conditions` | Categorical | Natural and street lighting conditions | `Daylight`, `Darkness - lights lit`, `Darkness - lights unlit`, `Darkness - no lighting` |
| `road_type` | Categorical | Configuration of road layout | `Single carriageway`, `Dual carriageway`, `Roundabout`, `One way street`, `Slip road` |
| `urban_or_rural_area` | Categorical | Urban density category | `Urban`, `Rural` |
| `junction_detail` | Categorical | Junction geometry | `Not at junction`, `T or staggered junction`, `Crossroads`, `Roundabout`, `More than 4 arms` |
| `number_of_vehicles` | Numerical | Count of vehicles involved in the incident | `1` to `5+` |
| `driver_age_band` | Categorical | Age band of primary driver | `18-25`, `26-35`, `36-55`, `56-65`, `65+` |
| `vehicle_type` | Categorical | Primary vehicle category | `Car`, `Motorcycle`, `Bus`, `Goods vehicle`, `Other` |
| **`accident_severity`** (Target) | Categorical | Injury severity classification | **`Slight`**, **`Serious`**, **`Fatal`** |

---

## 3. Working With Real-World Public Datasets

This pipeline is designed to be agnostic to the underlying dataset. You can easily plug in real-world public datasets by placing them into `data/raw/` and updating `config/config.yaml`.

Recommended open datasets:
1. **UK Road Safety Data (STATS19)**:
   - Source: [UK Government Department for Transport Open Data](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data)
   - License: Open Government Licence v3.0 (OGL)
   - Columns map directly to `accident_severity` (1: Fatal, 2: Serious, 3: Slight).
2. **US Accidents Dataset (2016 - Present)**:
   - Source: [Kaggle US Accidents](https://www.kaggle.com/datasets/sobhanmohammadpour/us-accidents)
   - License: CC BY-NC-SA 4.0
   - Severity scored from 1 to 4 based on traffic delay impact.
3. **Road Traffic Severity Dataset (Addis Ababa)**:
   - Source: [Kaggle Road Traffic Accidents](https://www.kaggle.com/datasets/saurabhshahane/road-traffic-accidents)
   - Targets: `Slight Injury`, `Serious Injury`, `Fatal Injury`.

---

## 4. Ethical Considerations and Disclaimer

> **IMPORTANT**:
> Statistical and machine learning predictions are strictly **decision-support estimates** intended for analytical research and safety auditing. They must **never** be treated as guaranteed outcomes or used as the sole operational basis for emergency response routing, insurance liability determination, or real-time road dispatch decisions.
