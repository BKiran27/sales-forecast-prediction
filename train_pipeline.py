"""
Standalone Production Training Pipeline for Sales Forecasting
Trains XGBoost, LightGBM, and Ensemble models with complete feature engineering.
Outputs serialized artifacts to models/ for immediate Streamlit & API inference.
"""

import os
import sys
import yaml
import joblib
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Ensure include and ui paths are available
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INCLUDE_DIR = os.path.join(BASE_DIR, "include")
UI_DIR = os.path.join(BASE_DIR, "ui")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, INCLUDE_DIR)
sys.path.insert(0, UI_DIR)

from include.utils.data_generator import RealisticSalesDataGenerator
from include.feature_engineering.feature_pipeline import FeatureEngineer
from ui.utils.ensemble_model_standalone import EnsembleModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def train_and_export():
    logger.info("==========================================================")
    logger.info("🚀 Starting End-to-End Sales Forecasting Training Pipeline")
    logger.info("==========================================================")
    
    # 1. Generate Realistic Data
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    models_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    for sub in ["xgboost", "lightgbm", "ensemble"]:
        os.makedirs(os.path.join(models_dir, sub), exist_ok=True)

    logger.info("Step 1: Generating realistic multi-store sales data...")
    generator = RealisticSalesDataGenerator(start_date="2022-01-01", end_date="2023-12-31")
    raw_output_dir = os.path.join(data_dir, "raw_partitions")
    os.makedirs(raw_output_dir, exist_ok=True)
    
    file_paths = generator.generate_sales_data(output_dir=raw_output_dir)
    
    # Load and consolidate sales files
    sales_dfs = []
    max_files = min(len(file_paths['sales']), 120)
    for sales_file in file_paths['sales'][:max_files]:
        sales_dfs.append(pd.read_parquet(sales_file))
    sales_df = pd.concat(sales_dfs, ignore_index=True)
    
    # Store-level daily aggregation
    daily_sales = sales_df.groupby(['date', 'store_id']).agg({
        'revenue': 'sum',
        'quantity_sold': 'sum',
        'profit': 'sum',
        'discount_percent': 'mean'
    }).reset_index().rename(columns={'revenue': 'sales'})
    
    # Merge traffic if available
    if file_paths.get('customer_traffic'):
        traffic_dfs = [pd.read_parquet(f) for f in file_paths['customer_traffic'][:15]]
        traffic_df = pd.concat(traffic_dfs, ignore_index=True)
        traffic_summary = traffic_df.groupby(['date', 'store_id']).agg({
            'customer_traffic': 'sum',
            'is_holiday': 'max'
        }).reset_index()
        daily_sales = daily_sales.merge(traffic_summary, on=['date', 'store_id'], how='left')
    else:
        daily_sales['customer_traffic'] = 500
        daily_sales['is_holiday'] = 0
        
    daily_sales['customer_traffic'] = daily_sales['customer_traffic'].fillna(500)
    daily_sales['is_holiday'] = daily_sales['is_holiday'].fillna(0).astype(int)
    daily_sales['has_promotion'] = (daily_sales['discount_percent'] > 0).astype(int)
    daily_sales['date'] = pd.to_datetime(daily_sales['date'])
    
    # Save a CSV sample for Streamlit UI upload testing
    sample_csv_path = os.path.join(data_dir, "sample_sales_data.csv")
    sample_export = daily_sales[daily_sales['store_id'] == 'store_001'][['date', 'sales', 'store_id']].copy()
    sample_export.to_csv(sample_csv_path, index=False)
    logger.info(f"Saved sample CSV for inference upload at: {sample_csv_path}")
    
    # 2. Feature Engineering
    logger.info("Step 2: Performing Feature Engineering Pipeline...")
    config_path = os.path.join(INCLUDE_DIR, "config", "ml_config.yaml")
    fe = FeatureEngineer(config_path=config_path)
    
    df_features = fe.create_all_features(
        daily_sales,
        target_col='sales',
        date_col='date',
        group_cols=['store_id'],
        categorical_cols=['store_id']
    )
    
    df_features = df_features.dropna(subset=['sales']).reset_index(drop=True)
    df_sorted = df_features.sort_values('date').reset_index(drop=True)
    
    # Train / Test split chronologically
    train_size = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:train_size].copy()
    test_df = df_sorted.iloc[train_size:].copy()
    
    exclude_cols = ['date', 'sales', 'store_id']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    # Encode categoricals if any
    encoders = {}
    if 'store_id' in train_df.columns:
        le = LabelEncoder()
        train_df['store_id_encoded'] = le.fit_transform(train_df['store_id'].astype(str))
        test_df['store_id_encoded'] = le.transform(test_df['store_id'].astype(str))
        encoders['store_id'] = le
        feature_cols.append('store_id_encoded')
        
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df['sales'].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df['sales'].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scalers = {'features': scaler}
    
    logger.info(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
    logger.info(f"Number of engineered features: {len(feature_cols)}")
    
    # 3. Train Models
    logger.info("Step 3: Training XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_preds = xgb_model.predict(X_test_scaled)
    
    logger.info("Step 4: Training LightGBM Regressor...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=150,
        num_leaves=31,
        learning_rate=0.08,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(X_train_scaled, y_train)
    lgb_preds = lgb_model.predict(X_test_scaled)
    
    logger.info("Step 5: Creating Weighted Ensemble Model...")
    ensemble = EnsembleModel(
        models={'xgboost': xgb_model, 'lightgbm': lgb_model},
        weights={'xgboost': 0.5, 'lightgbm': 0.5}
    )
    ens_preds = ensemble.predict(X_test_scaled)
    
    # 4. Evaluation & Benchmarking
    def calc_metrics(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
        return {'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape, 'R2': r2}
        
    metrics = {
        'XGBoost': calc_metrics(y_test, xgb_preds),
        'LightGBM': calc_metrics(y_test, lgb_preds),
        'Ensemble': calc_metrics(y_test, ens_preds)
    }
    
    logger.info("==========================================================")
    logger.info("📊 MODEL EVALUATION BENCHMARK RESULTS")
    logger.info("==========================================================")
    summary_df = pd.DataFrame(metrics).T
    print(summary_df.to_string())
    logger.info("==========================================================")
    
    # 5. Export Artifacts for Streamlit & API
    logger.info("Step 6: Serializing model artifacts to models/ directory...")
    joblib.dump(scalers, os.path.join(models_dir, "scalers.pkl"))
    joblib.dump(encoders, os.path.join(models_dir, "encoders.pkl"))
    joblib.dump(feature_cols, os.path.join(models_dir, "feature_cols.pkl"))
    
    # Save individual & ensemble models
    joblib.dump(xgb_model, os.path.join(models_dir, "xgboost", "xgboost_model.pkl"))
    joblib.dump(xgb_model, os.path.join(models_dir, "xgboost_model.pkl"))
    
    joblib.dump(lgb_model, os.path.join(models_dir, "lightgbm", "lightgbm_model.pkl"))
    joblib.dump(lgb_model, os.path.join(models_dir, "lightgbm_model.pkl"))
    
    joblib.dump(ensemble, os.path.join(models_dir, "ensemble", "ensemble_model.pkl"))
    joblib.dump(ensemble, os.path.join(models_dir, "ensemble_model.pkl"))
    
    logger.info("✅ All models, scalers, encoders, and feature column lists exported successfully!")
    logger.info("The application is now 100% ready for local execution and Streamlit Cloud deployment.")

if __name__ == "__main__":
    train_and_export()
