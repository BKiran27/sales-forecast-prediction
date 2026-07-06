import os
import argparse
import pandas as pd
import numpy as np
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

def calculate_single_holiday_distances(date, holiday_list):
    # Calculate days since last holiday and days to next holiday for a single date
    ts = pd.Timestamp(date)
    
    # Prior holidays
    past_holidays = [h for h in holiday_list if h <= ts]
    days_since = (ts - max(past_holidays)).days if past_holidays else 365
    
    # Future holidays
    future_holidays = [h for h in holiday_list if h >= ts]
    days_to = (min(future_holidays) - ts).days if future_holidays else 365
    
    return int(days_since), int(days_to)

def forecast_sales(start_date, end_date, store_id, category, promotion_active=0, competitor_discount=0, weather_event="Clear", is_stockout=0, data_path="data/raw_sales.csv", model_dir="models"):
    target_start = pd.Timestamp(start_date)
    target_end = pd.Timestamp(end_date)
    
    if target_start > target_end:
        raise ValueError("Start date must be before or equal to end date.")
        
    print(f"Loading trained model, encoders, and history for {store_id} - {category}...")
    
    # Load model artifacts
    model_path = os.path.join(model_dir, "sales_forecast_model.joblib")
    features_path = os.path.join(model_dir, "feature_names.joblib")
    encoder_path = os.path.join(model_dir, "cat_encoder.joblib")
    
    if not all(os.path.exists(p) for p in [model_path, features_path, encoder_path]):
        raise FileNotFoundError("Model artifacts not found. Please run features.py and train.py first.")
        
    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    encoder = joblib.load(encoder_path)
    
    # Load historical sales data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Historical sales data not found at {data_path}. Run data_generator.py first.")
        
    raw_df = pd.read_csv(data_path)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    
    history_df = raw_df[(raw_df["Store_ID"] == store_id) & (raw_df["Product_Category"] == category)].copy()
    history_df = history_df.sort_values(by="Date").reset_index(drop=True)
    
    max_history_date = history_df["Date"].max()
    
    sim_start = max_history_date + pd.Timedelta(days=1)
    sim_end = max(target_end, sim_start)
    
    sim_dates = pd.date_range(start=sim_start, end=sim_end, freq='D')
    
    # Prepare holiday list for simulation years
    sim_years = list(range(sim_start.year, sim_end.year + 2)) # Extra buffer year
    holiday_list = get_holiday_list(sim_years)
    
    # Seed history list
    history_records = history_df[["Date", "Sales"]].to_dict('records')
    for r in history_records:
        r["Date"] = pd.Timestamp(r["Date"])
        
    print(f"Running recursive daily forecasting from {sim_start.strftime('%Y-%m-%d')} to {sim_end.strftime('%Y-%m-%d')}...")
    
    def get_sales_on_date(d):
        for r in reversed(history_records):
            if r["Date"] == d:
                return r["Sales"]
        return 0
        
    for date in sim_dates:
        day_of_year = date.dayofyear
        
        # Temperature (seasonal estimation)
        temp_base = 65 + 20 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        temperature = round(temp_base + np.random.normal(0, 1.5), 1)
        
        is_holiday = 1 if date in holiday_list else 0
        
        # Holiday distances
        days_since, days_to = calculate_single_holiday_distances(date, holiday_list)
        
        # Lags
        lag_7 = get_sales_on_date(date - pd.Timedelta(days=7))
        lag_14 = get_sales_on_date(date - pd.Timedelta(days=14))
        lag_30 = get_sales_on_date(date - pd.Timedelta(days=30))
        
        # Rolling stats (mean & std of past sales, shifted by 1 day)
        roll_7_vals = [get_sales_on_date(date - pd.Timedelta(days=i)) for i in range(1, 8)]
        roll_30_vals = [get_sales_on_date(date - pd.Timedelta(days=i)) for i in range(1, 31)]
        
        roll_mean_7 = np.mean(roll_7_vals)
        roll_mean_30 = np.mean(roll_30_vals)
        roll_std_7 = np.std(roll_7_vals)
        roll_std_30 = np.std(roll_30_vals)
        
        # OHE Encoding mapping
        cat_df = pd.DataFrame([[store_id, category, weather_event]], columns=["Store_ID", "Product_Category", "Weather_Event"])
        encoded_cats = encoder.transform(cat_df)
        encoded_cols = encoder.get_feature_names_out(["Store_ID", "Product_Category", "Weather_Event"])
        encoded_dict = dict(zip(encoded_cols, encoded_cats[0]))
        
        # Features map
        feat_dict = {
            "Temperature": temperature,
            "Is_Holiday": is_holiday,
            "Promotion_Active": promotion_active,
            "Competitor_Discount_Active": competitor_discount,
            "Is_Stockout": is_stockout,
            "Year": date.year,
            "Month": date.month,
            "Day": date.day,
            "DayOfWeek": date.dayofweek,
            "DayOfYear": date.dayofyear,
            "Is_Weekend": 1 if date.dayofweek >= 5 else 0,
            "Month_Sin": np.sin(2 * np.pi * date.month / 12),
            "Month_Cos": np.cos(2 * np.pi * date.month / 12),
            "DayOfWeek_Sin": np.sin(2 * np.pi * date.dayofweek / 7),
            "DayOfWeek_Cos": np.cos(2 * np.pi * date.dayofweek / 7),
            "DayOfYear_Sin": np.sin(2 * np.pi * date.dayofyear / 365.25),
            "DayOfYear_Cos": np.cos(2 * np.pi * date.dayofyear / 365.25),
            "Days_Since_Last_Holiday": days_since,
            "Days_To_Next_Holiday": days_to,
            "Sales_Lag_7": lag_7,
            "Sales_Lag_14": lag_14,
            "Sales_Lag_30": lag_30,
            "Sales_Roll_Mean_7": roll_mean_7,
            "Sales_Roll_Mean_30": roll_mean_30,
            "Sales_Roll_Std_7": roll_std_7,
            "Sales_Roll_Std_30": roll_std_30,
        }
        feat_dict.update(encoded_dict)
        
        X_df = pd.DataFrame([feat_dict])
        X_df = X_df[feature_cols] # Align cols
        
        # Model Prediction
        if is_stockout == 1:
            pred_sales = 0
        else:
            pred_sales = model.predict(X_df)[0]
            pred_sales = max(0, round(pred_sales))
            
        history_records.append({
            "Date": date,
            "Sales": pred_sales
        })
        
    # Extract results
    forecast_results = []
    for r in history_records:
        if target_start <= r["Date"] <= target_end:
            d_val = r["Date"]
            forecast_results.append({
                "Date": d_val.strftime("%Y-%m-%d"),
                "Day_of_Week": d_val.strftime("%A"),
                "Promo_Active": promotion_active,
                "Competitor_Promo": competitor_discount,
                "Weather": weather_event,
                "Stockout": is_stockout,
                "Forecasted_Sales": r["Sales"]
            })
            
    return pd.DataFrame(forecast_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Forecast CLI for Scaled XGBoost Model.")
    parser.add_argument("--start-date", type=str, default="2026-01-01", help="Start date of forecast (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2026-01-07", help="End date of forecast (YYYY-MM-DD)")
    
    stores_choices = ["Store_1", "Store_2", "Store_3", "Store_4", "Store_5"]
    parser.add_argument("--store", type=str, default="Store_1", choices=stores_choices, help="Store ID")
    
    cat_choices = ["Electronics", "Apparel", "Grocery", "Home_Kitchen", "Beauty_Personal_Care"]
    parser.add_argument("--category", type=str, default="Electronics", choices=cat_choices, help="Product Category")
    
    parser.add_argument("--promo", type=int, default=0, choices=[0, 1], help="Promotion active (0=No, 1=Yes)")
    parser.add_argument("--competitor-promo", type=int, default=0, choices=[0, 1], help="Competitor discount active (0=No, 1=Yes)")
    parser.add_argument("--weather", type=str, default="Clear", choices=["Clear", "Rain", "Snow"], help="Weather event")
    parser.add_argument("--stockout", type=int, default=0, choices=[0, 1], help="Force stockout condition (0=No, 1=Yes)")
    
    args = parser.parse_args()
    
    try:
        results = forecast_sales(
            start_date=args.start_date,
            end_date=args.end_date,
            store_id=args.store,
            category=args.category,
            promotion_active=args.promo,
            competitor_discount=args.competitor_promo,
            weather_event=args.weather,
            is_stockout=args.stockout
        )
        
        print("\n" + "="*70)
        print(f" ADVANCED FORECAST REPORT FOR {args.store.upper()} - {args.category.upper()}")
        print("="*70)
        print(results.to_string(index=False))
        print("="*70)
        total_sales = results["Forecasted_Sales"].sum()
        avg_sales = results["Forecasted_Sales"].mean()
        print(f"Total Forecasted Sales: {total_sales} units")
        print(f"Average Daily Sales:    {avg_sales:.1f} units")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"Error during forecasting: {e}")
