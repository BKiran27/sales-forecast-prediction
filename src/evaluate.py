import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(test_path="data/test_features.csv", model_dir="models", plots_dir="plots"):
    print("Loading test features and model...")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test features not found at {test_path}. Run features.py first.")
        
    test_df = pd.read_csv(test_path)
    
    # Load model and features
    model_path = os.path.join(model_dir, "sales_forecast_model.joblib")
    features_path = os.path.join(model_dir, "feature_names.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(features_path):
        raise FileNotFoundError("Trained model or feature list not found. Run train.py first.")
        
    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    
    # Split features and target
    X_test = test_df[feature_cols]
    y_test = test_df["Sales"]
    
    # Predict
    print("Generating predictions on test set...")
    test_preds = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, test_preds)
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    r2 = r2_score(y_test, test_preds)
    
    # Mean Absolute Percentage Error (avoid division by zero)
    y_test_non_zero = y_test.copy()
    y_test_non_zero[y_test_non_zero == 0] = 1 # fallback
    mape = np.mean(np.abs((y_test - test_preds) / y_test_non_zero)) * 100
    
    print("\n" + "="*40)
    print("           EVALUATION METRICS")
    print("="*40)
    print(f"Mean Absolute Error (MAE):      {mae:.2f} units")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} units")
    print(f"Mean Absolute % Error (MAPE):   {mape:.2f}%")
    print(f"R-squared (R2) Score:           {r2:.4f}")
    print("="*40 + "\n")
    
    # Save metrics to file
    os.makedirs(plots_dir, exist_ok=True)
    with open(os.path.join(plots_dir, "metrics.txt"), "w") as f:
        f.write("="*40 + "\n")
        f.write("           EVALUATION METRICS\n")
        f.write("="*40 + "\n")
        f.write(f"Mean Absolute Error (MAE):      {mae:.2f} units\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f} units\n")
        f.write(f"Mean Absolute % Error (MAPE):   {mape:.2f}%\n")
        f.write(f"R-squared (R2) Score:           {r2:.4f}\n")
        f.write("="*40 + "\n")
    
    # Add predictions back to test dataframe for visualization
    test_df["Predicted_Sales"] = test_preds
    
    # Plot 1: Actual vs Predicted Time-Series (Sample: Store_1, Apparel)
    print("Creating time-series evaluation plot...")
    plt.figure(figsize=(14, 6))
    
    sample_df = test_df[(test_df["Store_ID_Store_1"] == 1) & (test_df["Product_Category_Apparel"] == 1)].copy()
    sample_df["Date"] = pd.to_datetime(sample_df["Date"])
    sample_df = sample_df.sort_values(by="Date")
    
    plt.plot(sample_df["Date"], sample_df["Sales"], label="Actual Sales", color="#1f77b4", alpha=0.8, linewidth=1.5)
    plt.plot(sample_df["Date"], sample_df["Predicted_Sales"], label="Predicted Sales", color="#ff7f0e", linestyle="--", alpha=0.9, linewidth=1.5)
    
    plt.title("Actual vs Predicted Sales Forecast (Store_1 - Apparel)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sales Units", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    plot_ts_path = os.path.join(plots_dir, "eval_actual_vs_predicted.png")
    plt.savefig(plot_ts_path, dpi=150)
    plt.close()
    
    # Plot 2: Feature Importance
    print("Creating feature importance plot...")
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(15) # Show top 15
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feat_imp_df, hue="Feature", palette="viridis", legend=False)
    plt.title("Top 15 Feature Importances (XGBoost Regressor)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Relative Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.tight_layout()
    
    plot_fi_path = os.path.join(plots_dir, "eval_feature_importance.png")
    plt.savefig(plot_fi_path, dpi=150)
    plt.close()
    
    # Plot 3: Residuals Analysis
    print("Creating residuals analysis plot...")
    residuals = y_test - test_preds
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Actual vs Predicted Scatter
    sns.scatterplot(x=y_test, y=test_preds, ax=axes[0], alpha=0.4, color="#4caf50")
    # Diagonal baseline
    max_val = max(y_test.max(), test_preds.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--', lw=2)
    axes[0].set_title("Actual vs. Predicted Sales Scatter", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Actual Sales", fontsize=10)
    axes[0].set_ylabel("Predicted Sales", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Right: Residuals Distribution
    sns.histplot(residuals, kde=True, ax=axes[1], color="#9c27b0")
    axes[1].axvline(0, color='r', linestyle='--', lw=2)
    axes[1].set_title("Error (Residuals) Distribution", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Residual (Actual - Predicted)", fontsize=10)
    axes[1].set_ylabel("Density", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_res_path = os.path.join(plots_dir, "eval_residuals_analysis.png")
    plt.savefig(plot_res_path, dpi=150)
    plt.close()
    
    print(f"Evaluation complete. Saved plots & metrics to the '{plots_dir}/' directory.")

if __name__ == "__main__":
    evaluate_model()
