# 🚀 Real-Time End-to-End Sales Forecasting Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=BKiran27/sales-forecast-prediction&branch=main&mainModule=streamlit_app.py)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Apache Airflow](https://img.shields.io/badge/Orchestration-Apache%20Airflow-017CEE.svg)](https://airflow.apache.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-EB680B.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, end-to-end Machine Learning and Data Engineering sales forecasting platform inspired by CodeWithYu / Yusuf Ganiyu's Astronomer architecture ([`airscholar/astro-salesforecast`](https://github.com/airscholar/astro-salesforecast)). 

This platform features automated pipeline orchestration with **Apache Airflow**, multi-model ML architectures (**XGBoost**, **LightGBM**, and **Weighted Ensemble**), and an interactive **Streamlit** dashboard configured for **1-click zero-configuration live deployment on Streamlit Community Cloud**.

---

## ⚡ 1-Click Live Deployment (Streamlit Community Cloud)

Deploy and launch this live app on the web directly from this GitHub repository:

👉 **[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=BKiran27/sales-forecast-prediction&branch=main&mainModule=streamlit_app.py)**  
**Direct URL**: [https://share.streamlit.io/deploy?repository=BKiran27/sales-forecast-prediction&branch=main&mainModule=streamlit_app.py](https://share.streamlit.io/deploy?repository=BKiran27/sales-forecast-prediction&branch=main&mainModule=streamlit_app.py)

> **No setup required**: The repository comes pre-bundled with trained ML models (`models/`), scalers, encoders, and sample data. When deployed, Streamlit Community Cloud automatically launches without needing Docker, Airflow, or MLflow servers!

---

## 🌟 Key Highlights

- **Dual-Mode Inference Architecture**:
  - **Standalone / Streamlit Cloud Mode**: Ready out-of-the-box! Bundled with pre-trained models (`models/`), scalers, encoders, and sample data. Runs error-free on Streamlit Community Cloud or any local terminal.
  - **Full Airflow / MLOps Mode**: Complete Astronomer DAG orchestration (`dags/sales_forecast_training.py`) covering synthetic data generation, Pandera schema validation, feature engineering, MLflow model tracking, and Model Registry deployment.
- **Advanced Machine Learning**:
  - **XGBoost Regressor**: Gradient boosting capturing complex non-linear retail patterns ($R^2 = 0.9476$).
  - **LightGBM Regressor**: Fast, leaf-wise gradient boosting optimized for tabular data ($R^2 = 0.8561$).
  - **Weighted Dynamic Ensemble**: Blends predictions from XGBoost and LightGBM to minimize variance and forecast error ($R^2 = 0.9157$).
- **Rich Feature Engineering**:
  - **Calendar & Cyclical**: Day of week, month, day, quarter, week of year, cyclical $\sin$ and $\cos$ encodings.
  - **Lags & Moving Statistics**: 1, 2, 3, 7, 14, 21, and 30-day sales lags; 3, 7, 14, 21, and 30-day moving averages, standard deviations, and min/max/median bounds.
  - **Exogenous Variables**: Holiday markers (via `holidays` library), promotional discount percentages, and customer foot traffic.
- **Interactive Streamlit Web Dashboard**:
  - **Active / Sample Data Scenario Generator**: Pre-loaded with realistic retail sales data with 1-click custom scenario simulations.
  - **Upload Custom CSV**: Upload your store's sales dataset (`date`, `sales`, and optional `store_id`) for dynamic forecasts.
  - **Manual 7-Day Entry**: Quickly simulate upcoming sales with an intuitive daily input grid.
  - **Forecast Horizon**: Slide between 1 to 90 days ahead with 90% confidence bands.
  - **Interactive Plotly Visualizations**: Drill into historical trends and predicted curves with hover tooltips and zoom controls.
  - **One-Click CSV Export**: Download the generated predictions and confidence bounds directly to CSV.

---

## 📊 Model Evaluation Benchmark Results

Trained on realistic daily retail transactions across multiple stores with chronological train/test split:

| Model | RMSE | MAE | MAPE (%) | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost Regressor** | **23.71** | **11.92** | **8.45%** | **0.9476** |
| **LightGBM Regressor** | 39.28 | 21.11 | 14.20% | 0.8561 |
| **Ensemble (Weighted Blend)** | 30.06 | 15.13 | 10.16% | 0.9157 |

---

## 🏗️ Architecture & Directory Structure

```text
sales-forecast-prediction/
├── dags/                                 # Apache Airflow DAGs
│   ├── .airflowignore                    # Airflow ignore rules
│   └── sales_forecast_training.py        # End-to-end Airflow ETL & ML DAG
├── include/                              # Production pipeline modules
│   ├── config/
│   │   ├── ml_config.yaml                # Model, feature & path configuration
│   │   └── ml_config_local.yaml          # Local fallback configuration
│   ├── data_validation/
│   │   └── validators.py                 # Pandera schema validation
│   ├── feature_engineering/
│   │   └── feature_pipeline.py           # Feature extraction & scaling
│   ├── ml_models/
│   │   ├── advanced_ensemble.py          # Dynamic stacking & ensembling
│   │   ├── diagnostics.py                # Model diagnostics & residual analysis
│   │   ├── ensemble_model.py             # Airflow ensemble runner
│   │   ├── model_comparison.py           # MLflow model comparison
│   │   ├── model_visualization.py        # Feature importance & evaluation plots
│   │   └── train_models.py               # Model training implementations
│   ├── model_serving/
│   │   └── inference_api.py              # FastAPI / serving endpoint
│   └── utils/
│       ├── data_generator.py             # Realistic multi-store sales simulator
│       ├── metrics.py                    # RMSE, MAE, MAPE, R2 metrics
│       ├── mlflow_utils.py               # MLflow tracking utilities
│       └── parquet_validator.py          # Parquet schema verification
├── models/                               # Serialized ML artifacts (pre-packaged)
│   ├── scalers.pkl                       # Feature StandardScaler
│   ├── encoders.pkl                      # Categorical LabelEncoder
│   ├── feature_cols.pkl                  # Exact ordered feature list
│   ├── xgboost_model.pkl                 # Trained XGBoost artifact
│   ├── lightgbm_model.pkl                # Trained LightGBM artifact
│   └── ensemble_model.pkl                # Trained Ensemble artifact
├── data/
│   └── sample_sales_data.csv             # Ready-to-use sample dataset for UI testing
├── ui/                                   # Streamlit application components
│   ├── __init__.py                       # UI package marker
│   ├── inference_app.py                  # Full inference web application
│   └── utils/
│       ├── __init__.py                   # UI utils package marker
│       ├── simple_model_loader.py        # Dual-mode loader (Local files + MLflow)
│       ├── simple_predictor.py           # Feature transformer & inference engine
│       └── ensemble_model_standalone.py  # Standalone ensemble wrapper
├── streamlit_app.py                      # Root entrypoint for Streamlit Community Cloud
├── train_pipeline.py                     # Standalone model training & export script
├── requirements.txt                      # Production dependencies
├── Dockerfile                            # Astronomer Airflow container definition
├── docker-compose.override.yml           # Airflow service overrides
├── airflow_settings.yaml                 # Airflow connections & pools
└── README.md                             # Project documentation
```

---

## 💻 Running the App

### Option 1: Run Directly via GitHub URL (No Cloning Required)
You can run the app directly through Git on any terminal with Streamlit installed:

```bash
streamlit run https://raw.githubusercontent.com/BKiran27/sales-forecast-prediction/main/streamlit_app.py
```

---

### Option 2: Clone and Run Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/BKiran27/sales-forecast-prediction.git
   cd sales-forecast-prediction
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux / macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the dashboard**:
   ```bash
   streamlit run streamlit_app.py
   ```
   Open your browser at `http://localhost:8501`.

---

## 🧪 Retraining Models Locally

To regenerate multi-store data, re-run feature engineering, train XGBoost, LightGBM, and Ensemble models, and update the serialized artifacts in `models/`:

```bash
python train_pipeline.py
```

---

## 🐳 Full Orchestration with Astronomer / Apache Airflow

If you have Docker and the Astronomer CLI (`astro`) installed, you can launch the complete enterprise MLOps platform:

```bash
# Start Astronomer Airflow stack
astro dev start

# Airflow UI: http://localhost:8080 (admin / admin)
# MLflow UI:  http://localhost:5001
# MinIO UI:   http://localhost:9001 (minioadmin / minioadmin)
```

Trigger DAG `sales_forecast_training` in the Airflow UI to run the automated lifecycle:
1. `generate_sales_data` $\rightarrow$ Simulates daily transactions across 10 retail stores.
2. `validate_raw_data` $\rightarrow$ Validates schemas and null bounds using Pandera.
3. `engineer_features` $\rightarrow$ Generates lag features, cyclical encodings, and moving averages.
4. `train_xgboost` & `train_lightgbm` $\rightarrow$ Trains models in parallel with MLflow parameter & metric tracking.
5. `evaluate_and_ensemble` $\rightarrow$ Evaluates test performance and blends base models into a weighted ensemble.
6. `deploy_best_model` $\rightarrow$ Registers the champion model into the MLflow Model Registry.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
