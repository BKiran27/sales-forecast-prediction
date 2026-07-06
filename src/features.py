import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib

def get_holiday_list(years):
    holidays = []
    for year in years:
        holidays.extend([
            pd.Timestamp(f"{year}-01-01"),  # New Year's Day
            pd.Timestamp(f"{year}-07-04"),  # Independence Day
            pd.Timestamp(f"{year}-11-25"),  # Thanksgiving approx
            pd.Timestamp(f"{year}-11-26"),  # Black Friday
            pd.Timestamp(f"{year}-11-27"),  
            pd.Timestamp(f"{year}-12-24"),  # Christmas Eve
            pd.Timestamp(f"{year}-12-25"),  # Christmas Day
            pd.Timestamp(f"{year}-12-31"),  # New Year's Eve
        ])
    return sorted(list(set(holidays)))

def calculate_holiday_distances(df_dates, holiday_list):
    # Optimizing holiday proximity calculations
    h_arr = np.array([h.value for h in holiday_list]) # Nanosecond timestamps
    d_arr = df_dates.astype('int64').values[:, None]
    
    # Absolute difference matrix
    diff = d_arr - h_arr
    
    # Days since last holiday: minimum positive diff (converted to days)
    # We find positive differences (days in past)
    diff_past = np.where(diff >= 0, diff, np.inf)
    idx_past = np.argmin(diff_past, axis=1)
    min_diff_past = diff_past[np.arange(len(df_dates)), idx_past]
    days_since = np.where(min_diff_past == np.inf, 365, min_diff_past / (10**9 * 3600 * 24))
    
    # Days to next holiday: minimum negative diff in absolute terms (converted to days)
    diff_future = np.where(diff <= 0, -diff, np.inf)
    idx_future = np.argmin(diff_future, axis=1)
    min_diff_future = diff_future[np.arange(len(df_dates)), idx_future]
    days_to = np.where(min_diff_future == np.inf, 365, min_diff_future / (10**9 * 3600 * 24))
    
    return days_since.astype(int), days_to.astype(int)

def engineer_features(data_path="data/raw_sales.csv", output_dir="data"):
    print("Loading raw sales data...")
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Sort chronologically
    df = df.sort_values(by=["Store_ID", "Product_Category", "Date"]).reset_index(drop=True)
    
    # 1. Date Components & Cyclical Encodings
    print("Extracting datetime features and cyclical encodings...")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Is_Weekend"] = (df["DayOfWeek"] >= 5).astype(int)
    
    # Cyclical transformations using sine & cosine to capture circularity
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DayOfWeek_Sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_Cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfYear_Sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
    df["DayOfYear_Cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)
    
    # 2. Holiday Proximity Features
    print("Calculating proximity to holidays...")
    unique_years = df["Year"].unique()
    holiday_list = get_holiday_list(unique_years)
    
    # Calculate once for unique dates to be faster
    unique_dates = pd.Series(df["Date"].unique()).sort_values()
    days_since, days_to = calculate_holiday_distances(unique_dates, holiday_list)
    
    dist_map = pd.DataFrame({
        "Date": unique_dates,
        "Days_Since_Last_Holiday": days_since,
        "Days_To_Next_Holiday": days_to
    })
    df = df.merge(dist_map, on="Date", how="left")
    
    # 3. Lag Features
    print("Generating lag features...")
    # Shift sales grouped by store and product category
    df["Sales_Lag_7"] = df.groupby(["Store_ID", "Product_Category"])["Sales"].shift(7)
    df["Sales_Lag_14"] = df.groupby(["Store_ID", "Product_Category"])["Sales"].shift(14)
    df["Sales_Lag_30"] = df.groupby(["Store_ID", "Product_Category"])["Sales"].shift(30)
    
    # 4. Rolling Statistics (Mean & Std Dev)
    # Note: Shifted by 1 to prevent data leakage of today's target
    print("Generating rolling average and standard deviation features...")
    grouped = df.groupby(["Store_ID", "Product_Category"])["Sales"].shift(1)
    
    df["Sales_Roll_Mean_7"] = grouped.rolling(window=7).mean().reset_index(0, drop=True)
    df["Sales_Roll_Mean_30"] = grouped.rolling(window=30).mean().reset_index(0, drop=True)
    
    df["Sales_Roll_Std_7"] = grouped.rolling(window=7).std().reset_index(0, drop=True)
    df["Sales_Roll_Std_30"] = grouped.rolling(window=30).std().reset_index(0, drop=True)
    
    # Fill standard deviation NaNs (which can occur on constant regions or early dates) with 0
    df["Sales_Roll_Std_7"] = df["Sales_Roll_Std_7"].fillna(0)
    df["Sales_Roll_Std_30"] = df["Sales_Roll_Std_30"].fillna(0)
    
    # Drop rows with NaN values resulting from lags/rolling windows
    print("Dropping initial NaN rows...")
    df = df.dropna().reset_index(drop=True)
    
    # 5. One-Hot Encoding for Categorical Variables
    print("Encoding categorical variables...")
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    categorical_cols = ["Store_ID", "Product_Category", "Weather_Event"]
    
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
    
    df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)
    df_final = pd.concat([df, df_encoded], axis=1)
    df_final = df_final.drop(columns=categorical_cols)
    
    # Save categorical encoder
    os.makedirs("models", exist_ok=True)
    encoder_path = os.path.join("models", "cat_encoder.joblib")
    joblib.dump(encoder, encoder_path)
    print(f"Saved categorical encoder to {encoder_path}")
    
    # 6. Train/Test Split (Temporal)
    # Split: Train on 2021-2024; Test on 2025 (last full year)
    cutoff_date = pd.Timestamp("2025-01-01")
    
    train_df = df_final[df_final["Date"] < cutoff_date]
    test_df = df_final[df_final["Date"] >= cutoff_date]
    
    print("Temporal dataset split results:")
    print(f"  Train: {train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')} ({len(train_df)} rows)")
    print(f"  Test:  {test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')} ({len(test_df)} rows)")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, "train_features.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test_features.csv"), index=False)
        print(f"Saved feature-engineered train/test splits to {output_dir}/")
        
    return train_df, test_df, list(encoded_feature_names)

if __name__ == "__main__":
    engineer_features()
