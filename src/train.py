import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import joblib

def train_model(train_path="data/train_features.csv", model_dir="models"):
    print("Loading training features...")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training features not found at {train_path}. Run features.py first.")
        
    train_df = pd.read_csv(train_path)
    
    # Sort chronologically to make sure time series cross-validation is valid
    train_df = train_df.sort_values(by="Date").reset_index(drop=True)
    
    target_col = "Sales"
    non_feature_cols = ["Date", "Sales"]
    feature_cols = [col for col in train_df.columns if col not in non_feature_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    print(f"Training features shape: {X_train.shape}")
    
    # 1. Setup Time Series Cross-Validation Splitter
    # Ensures folds are selected in temporal order to mimic real-world forecasting
    tscv = TimeSeriesSplit(n_splits=5)
    
    # 2. Hyperparameter Grid
    param_dist = {
        "n_estimators": [150, 250, 350],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.06, 0.1],
        "subsample": [0.75, 0.85],
        "colsample_bytree": [0.75, 0.85],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.2]
    }
    
    # 3. Randomized Search CV
    print("Starting hyperparameter optimization with TimeSeriesSplit (5 folds)...")
    xgb = XGBRegressor(random_state=42, n_jobs=-1)
    
    # Run 10 iterations to balance thoroughness and execution speed
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=10,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    print("\n" + "="*40)
    print("      HYPERPARAMETER SEARCH RESULTS")
    print("="*40)
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV MAE score: {-search.best_score_:.2f} units")
    print("="*40 + "\n")
    
    # Retrieve the best model
    best_model = search.best_estimator_
    
    # Fit the best model on all training data
    print("Re-fitting the best model configuration on the full training set...")
    best_model.fit(X_train, y_train)
    
    # Save the optimized model and features list
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "sales_forecast_model.joblib")
    features_path = os.path.join(model_dir, "feature_names.joblib")
    
    joblib.dump(best_model, model_path)
    joblib.dump(feature_cols, features_path)
    
    print(f"Saved optimized model to {model_path}")
    print(f"Saved training features list to {features_path}")
    
    # Evaluate final training performance
    train_preds = best_model.predict(X_train)
    train_mae = np.mean(np.abs(y_train - train_preds))
    train_rmse = np.sqrt(np.mean((y_train - train_preds) ** 2))
    print(f"Final Train Metrics: MAE = {train_mae:.2f}, RMSE = {train_rmse:.2f}")

if __name__ == "__main__":
    train_model()
