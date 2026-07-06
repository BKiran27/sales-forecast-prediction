# Sales Forecast Prediction ML Project

An end-to-end Machine Learning pipeline in Python for forecasting daily sales of a retail store using **XGBoost Regressor**.

This project models daily sales across multiple stores and product categories. It features realistic synthetic data generation (incorporating trends, weekly/yearly seasonality, holidays, and promotions), custom feature engineering (with leak-proof lag and rolling average features), temporal dataset splitting, evaluation metrics/visualizations, and a command-line interface (CLI) to forecast future sales recursively.

## Project Structure

```text
sales-forecast-prediction/
├── data/                      # Data storage (generated and preprocessed CSVs)
│   ├── raw_sales.csv
│   ├── train_features.csv
│   └── test_features.csv
├── models/                    # Saved models and preprocessors (joblib files)
│   ├── cat_encoder.joblib
│   ├── feature_names.joblib
│   └── sales_forecast_model.joblib
├── plots/                     # Visualizations and metrics text reports
│   ├── raw_sales_trend.png
│   ├── eval_actual_vs_predicted.png
│   ├── eval_feature_importance.png
│   ├── eval_residuals_analysis.png
│   └── metrics.txt
├── src/                       # Source code scripts
│   ├── data_generator.py      # Synthetic daily sales generator
│   ├── features.py            # Feature engineering pipeline
│   ├── train.py               # Model training script (XGBoost)
│   ├── evaluate.py            # Performance evaluation and plotting
│   └── predict.py             # CLI for future recursive forecasting
├── requirements.txt           # Package dependencies
└── README.md                  # Project documentation
```

---

## Installation & Setup

1. **Clone or Navigate to the project directory**:
   ```bash
   cd C:\Users\basav\.gemini\antigravity\scratch\sales-forecast-prediction
   ```

2. **Set up a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Pipeline Execution Steps

Execute each script in the pipeline sequentially:

### 1. Generate Raw Data
Creates 3 years of daily sales records across 3 stores and 3 categories (Grocery, Apparel, Electronics):
```bash
python src/data_generator.py
```
*Outputs*: Saves raw dataset to `data/raw_sales.csv` and a sample trend plot to `plots/raw_sales_trend.png`.

### 2. Run Feature Engineering
Extracts temporal markers, builds lags and rolling averages, encodes categories, and splits data chronologically (avoiding data leakage):
```bash
python src/features.py
```
*Outputs*: Saves `data/train_features.csv`, `data/test_features.csv`, and serializes the categorical transformer to `models/cat_encoder.joblib`.

### 3. Train the Model
Trains the XGBoost Regressor model on the feature-engineered training set:
```bash
python src/train.py
```
*Outputs*: Serializes the trained model to `models/sales_forecast_model.joblib` and feature names list to `models/feature_names.joblib`.

### 4. Evaluate the Model
Validates model performance on unseen test data, prints metrics, and saves residual/importance charts:
```bash
python src/evaluate.py
```
*Outputs*: Writes evaluation metrics to `plots/metrics.txt` and saves visual diagnostic plots in the `plots/` directory.

### 5. Run the Forecast CLI
Forecast future sales dynamically for a given date range, store, and product category. It runs a **recursive autoregressive forecast** day-by-day to simulate future lag features:
```bash
python src/predict.py --start-date 2026-01-01 --end-date 2026-01-07 --store Store_1 --category Apparel --promo 0
```

#### CLI Parameters:
- `--start-date` (default: `2026-01-01`): Start date of forecast (YYYY-MM-DD)
- `--end-date` (default: `2026-01-07`): End date of forecast (YYYY-MM-DD)
- `--store` (choices: `Store_1`, `Store_2`, `Store_3`): Store ID
- `--category` (choices: `Electronics`, `Apparel`, `Grocery`): Product Category
- `--promo` (choices: `0`, `1`, default: `0`): Whether promotion is active during the forecast window

---

## Machine Learning Architecture Highlights

- **XGBoost Regressor**: Used for training due to its high efficiency and accuracy with structured tabular data.
- **Leakage Prevention**: Rolling mean and standard deviation features are calculated using historical sales *shifted by 1 day* (e.g. `shift(1).rolling()`) to ensure the model does not sneak peak at the target sales value of the day being predicted.
- **Temporal Splitting**: Traditional random splits destroy temporal structure in time series. We use a chronological cut-off (Train: Jan 2023 - June 2025; Test: July 2025 - Dec 2025) to evaluate realistic out-of-sample performance.
- **Autoregressive Multi-step Forecasting**: When predicting into the future (beyond the historical dates), actual sales do not exist to fill the lag features. The CLI recursively feeds today's prediction back into the history pool to compute lags and rolling averages for tomorrow's prediction.
