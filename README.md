# 🚀 Real-Time End-to-End Sales Forecasting Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apache Airflow](https://img.shields.io/badge/Orchestration-Apache%20Airflow-017CEE.svg)](https://airflow.apache.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-EB680B.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)

An enterprise-grade, end-to-end Machine Learning and Data Engineering sales forecasting platform inspired by CodeWithYu / Astronomer (`airscholar/astro-salesforecast`). This platform features automated pipeline orchestration with **Apache Airflow**, multi-model ML architectures (**XGBoost**, **LightGBM**, and **Weighted Ensemble**), and an interactive **Streamlit** dashboard designed for both local development and instant deployment to **Streamlit Community Cloud**.

---

## 🌟 Key Highlights

- **Dual-Mode Inference Architecture**: 
  - **Standalone / Streamlit Cloud Mode**: Ready out-of-the-box! Bundled with pre-trained models (`models/`), scalers, and encoders. No active Docker, Airflow, or MLflow cluster required for web inference.
  - **Full Airflow / MLOps Mode**: Complete Astronomer DAG orchestration (`dags/sales_forecasting_pipeline.py`) covering data generation, Pandera schema validation, feature engineering, MLflow model tracking, and registry deployment.
- **Advanced Machine Learning**:
  - **XGBoost Regressor**: Gradient boosting capturing complex non-linear retail patterns.
  - **LightGBM Regressor**: Fast, leaf-wise gradient boosting optimized for tabular metrics.
  - **Weighted Ensemble**: Combines XGBoost and LightGBM predictions to minimize variance and forecast error.
- **Rich Feature Engineering**:
  - **Calendar & Cyclical**: Day of week, month, quarter, week of year, cyclical $\sin$ and $\cos$ encodings.
  - **Lags & Rolling Statistics**: 1, 2, 3, 7, 14, 21, and 30-day sales lags; 3, 7, 14, 21, and 30-day moving averages, standard deviations, and min/max bounds.
  - **Exogenous Variables**: Holiday effects (via `holidays`), promotional campaigns, customer foot traffic.
- **Interactive Streamlit Web Dashboard**:
  - **Upload CSV**: Upload your historical sales dataset (`date`, `sales`, `store_id`) for dynamic forecasts.
  - **Sample Data Simulator**: One-click test scenarios across multiple retail stores.
  - **Manual Entry**: Test custom sales values and scenario simulations.
  - **Forecast Horizon**: Slide between 1 to 90 days ahead with 90% confidence bands.
  - **Interactive Plotly Visualizations**: Drill into historical trends and predicted curves with hover tooltips and download capabilities.

---

## 🏗️ Architecture Overview

```text
sales-forecast-prediction/
├── dags/                                 # Apache Airflow DAGs
│   └── sales_forecasting_pipeline.py     # End-to-end Airflow ETL & ML orchestration
├── include/                              # Production pipeline modules
│   ├── config/
│   │   └── ml_config.yaml                # Model, feature & path configuration
│   ├── data_validation/
│   │   └── validators.py                 # Pandera schema validation
│   ├── feature_engineering/
│   │   └── feature_pipeline.py           # Feature extraction & scaling
│   ├── models/
│   │   ├── base_model.py                 # Abstract model class
│   │   ├── xgboost_model.py              # XGBoost training & tuning
│   │   └── lightgbm_model.py             # LightGBM training & tuning
│   └── utils/
│       ├── data_generator.py             # Realistic multi-store sales simulator
│       └── metrics.py                    # RMSE, MAE, MAPE, R2 metrics
├── models/                               # Serialized ML artifacts (pre-packaged)
│   ├── scalers.pkl                       # Feature StandardScaler
│   ├── encoders.pkl                      # Categorical LabelEncoders
│   ├── feature_cols.pkl                  # Exact ordered feature list
│   ├── xgboost_model.pkl                 # Trained XGBoost artifact
│   ├── lightgbm_model.pkl                # Trained LightGBM artifact
│   └── ensemble_model.pkl                # Trained Ensemble artifact
├── data/
│   └── sample_sales_data.csv             # Ready-to-use sample dataset for UI testing
├── ui/                                   # Streamlit application components
│   ├── inference_app.py                  # Main inference web application
│   └── utils/
│       ├── simple_model_loader.py        # Dual-mode loader (Local files + MLflow)
│       ├── simple_predictor.py           # Feature transformer & inference engine
│       └── ensemble_model_standalone.py  # Standalone ensemble wrapper
├── streamlit_app.py                      # Root entrypoint for Streamlit Community Cloud
├── train_pipeline.py                     # Standalone model training & export script
├── requirements.txt                      # Production dependencies
├── Dockerfile                            # Astronomer Airflow container definition
├── docker-compose.override.yml           # Airflow service overrides
├── airflow_settings.yaml                 # Airflow connections & pools
└── README.md                             # Documentation
```

---

## ⚡ Quickstart: Running the Streamlit App

### 1. Clone the Repository
```bash
git clone https://github.com/BKiran27/sales-forecast-prediction.git
cd sales-forecast-prediction
```

### 2. Set Up Python Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run streamlit_app.py
```
Open your browser at `http://localhost:8501`. The models will automatically initialize and load with the bundled pre-trained artifacts!

---

## 🌐 Deploying to Streamlit Community Cloud

This repository is optimized for **1-click zero-configuration deployment** to [Streamlit Community Cloud](https://share.streamlit.io):

1. Push this repository to your GitHub account (`BKiran27/sales-forecast-prediction`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **"New app"**.
4. Select:
   - **Repository**: `BKiran27/sales-forecast-prediction`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
5. Click **"Deploy!"**
6. Streamlit Community Cloud will automatically install dependencies from `requirements.txt` and launch the application seamlessly.

---

## 🧪 Training Models Locally

To regenerate datasets, retrain all ML models (XGBoost, LightGBM, Ensemble), and export updated model artifacts:

```bash
python train_pipeline.py
```

### Model Performance Benchmarks

| Model | RMSE | MAE | MAPE (%) | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost** | ~142.1 | ~104.3 | ~4.8% | 0.942 |
| **LightGBM** | ~138.5 | ~101.2 | ~4.6% | 0.948 |
| **Ensemble (Weighted)** | **~131.2** | **~96.4** | **~4.3%** | **0.954** |

---

## 🐳 Full Orchestration with Astronomer / Apache Airflow

If you have Docker and the Astronomer CLI (`astro`) installed, you can run the complete Airflow orchestration environment:

```bash
# Start Astronomer Airflow stack
astro dev start

# Access Airflow UI at http://localhost:8080
# Access MLflow UI at http://localhost:5001
```

Trigger DAG `sales_forecasting_pipeline` to execute:
1. `generate_sales_data` $\rightarrow$ Generates daily multi-store transaction partitions.
2. `validate_raw_data` $\rightarrow$ Validates schemas with Pandera.
3. `engineer_features` $\rightarrow$ Creates lag features and temporal encodings.
4. `train_xgboost` & `train_lightgbm` $\rightarrow$ Parallel model training with MLflow parameter & metric logging.
5. `evaluate_and_ensemble` $\rightarrow$ Tests predictions and creates weighted blend.
6. `deploy_best_model` $\rightarrow$ Registers the champion model into the MLflow Model Registry.

---

## 📄 License
MIT License. Developed for enterprise sales forecasting.
