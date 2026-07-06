import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Resolve directories relative to app.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw_sales.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

def get_holiday_list(years):
    holidays = []
    for year in years:
        holidays.extend([
            pd.Timestamp(f"{year}-01-01"),
            pd.Timestamp(f"{year}-07-04"),
            pd.Timestamp(f"{year}-11-25"),
            pd.Timestamp(f"{year}-11-26"),
            pd.Timestamp(f"{year}-11-27"),
            pd.Timestamp(f"{year}-12-24"),
            pd.Timestamp(f"{year}-12-25"),
            pd.Timestamp(f"{year}-12-31"),
        ])
    return sorted(list(set(holidays)))

def calculate_single_holiday_distances(date, holiday_list):
    ts = pd.Timestamp(date)
    past_holidays = [h for h in holiday_list if h <= ts]
    days_since = (ts - max(past_holidays)).days if past_holidays else 365
    
    future_holidays = [h for h in holiday_list if h >= ts]
    days_to = (min(future_holidays) - ts).days if future_holidays else 365
    return int(days_since), int(days_to)

def run_recursive_forecast(start_date, end_date, store_id, category, promotion_active, competitor_discount, weather_event, is_stockout):
    # Check model files
    model_path = os.path.join(MODEL_DIR, "sales_forecast_model.joblib")
    features_path = os.path.join(MODEL_DIR, "feature_names.joblib")
    encoder_path = os.path.join(MODEL_DIR, "cat_encoder.joblib")
    
    if not all(os.path.exists(p) for p in [model_path, features_path, encoder_path]):
        raise FileNotFoundError("Trained ML model artifacts not found. Please train the model first.")
        
    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    encoder = joblib.load(encoder_path)
    
    # Load history to seed lags & rolling means
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Raw historical sales dataset not found.")
        
    raw_df = pd.read_csv(DATA_PATH)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    
    # Seed history for target store/category
    history_df = raw_df[(raw_df["Store_ID"] == store_id) & (raw_df["Product_Category"] == category)].copy()
    history_df = history_df.sort_values(by="Date").reset_index(drop=True)
    
    max_history_date = history_df["Date"].max()
    
    target_start = pd.Timestamp(start_date)
    target_end = pd.Timestamp(end_date)
    
    sim_start = max_history_date + pd.Timedelta(days=1)
    sim_end = max(target_end, sim_start)
    
    sim_dates = pd.date_range(start=sim_start, end=sim_end, freq='D')
    
    sim_years = list(range(sim_start.year, sim_end.year + 2))
    holiday_list = get_holiday_list(sim_years)
    
    history_records = history_df[["Date", "Sales"]].to_dict('records')
    for r in history_records:
        r["Date"] = pd.Timestamp(r["Date"])
        
    def get_sales_on_date(d):
        for r in reversed(history_records):
            if r["Date"] == d:
                return r["Sales"]
        return 0
        
    for date in sim_dates:
        day_of_year = date.dayofyear
        
        # Temperature: seasonal estimation
        temp_base = 65 + 20 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        temperature = round(temp_base + np.random.normal(0, 1.5), 1)
        
        is_holiday = 1 if date in holiday_list else 0
        days_since, days_to = calculate_single_holiday_distances(date, holiday_list)
        
        # Lags
        lag_7 = get_sales_on_date(date - pd.Timedelta(days=7))
        lag_14 = get_sales_on_date(date - pd.Timedelta(days=14))
        lag_30 = get_sales_on_date(date - pd.Timedelta(days=30))
        
        # Rolling averages (shifted 1 day)
        roll_7_vals = [get_sales_on_date(date - pd.Timedelta(days=i)) for i in range(1, 8)]
        roll_30_vals = [get_sales_on_date(date - pd.Timedelta(days=i)) for i in range(1, 31)]
        
        roll_mean_7 = np.mean(roll_7_vals)
        roll_mean_30 = np.mean(roll_30_vals)
        roll_std_7 = np.std(roll_7_vals)
        roll_std_30 = np.std(roll_30_vals)
        
        # Encode categoricals
        cat_df = pd.DataFrame([[store_id, category, weather_event]], columns=["Store_ID", "Product_Category", "Weather_Event"])
        encoded_cats = encoder.transform(cat_df)
        encoded_cols = encoder.get_feature_names_out(["Store_ID", "Product_Category", "Weather_Event"])
        encoded_dict = dict(zip(encoded_cols, encoded_cats[0]))
        
        # Assemble feature dict
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
        X_df = X_df[feature_cols] # Align
        
        if is_stockout == 1:
            pred_sales = 0
        else:
            pred_sales = model.predict(X_df)[0]
            pred_sales = max(0, round(pred_sales))
            
        history_records.append({
            "Date": date,
            "Sales": pred_sales
        })
        
    # Format and extract response dates
    results = []
    for r in history_records:
        if target_start <= r["Date"] <= target_end:
            d_val = r["Date"]
            is_hol = 1 if d_val in holiday_list else 0
            temp = round(65 + 20 * np.sin(2 * np.pi * (d_val.dayofyear - 100) / 365), 1)
            
            results.append({
                "date": d_val.strftime("%Y-%m-%d"),
                "day_of_week": d_val.strftime("%A"),
                "is_holiday": is_hol,
                "temp": temp,
                "forecasted_sales": int(r["Sales"])
            })
            
    df_res = pd.DataFrame(results)
    summary = {
        "total_sales": int(df_res["forecasted_sales"].sum()) if not df_res.empty else 0,
        "avg_sales": float(round(df_res["forecasted_sales"].mean(), 1)) if not df_res.empty else 0.0,
        "max_sales": int(df_res["forecasted_sales"].max()) if not df_res.empty else 0
    }
    
    return results, summary

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON request body"}), 400
            
        # Parse & Validate Inputs
        start_date = data.get("start_date", "2026-01-01")
        end_date = data.get("end_date", "2026-01-07")
        store_id = data.get("store_id", "Store_1")
        category = data.get("category", "Electronics")
        promotion_active = int(data.get("promotion_active", 0))
        competitor_discount = int(data.get("competitor_discount", 0))
        weather_event = data.get("weather_event", "Clear")
        is_stockout = int(data.get("is_stockout", 0))
        
        # Validate values
        if store_id not in ["Store_1", "Store_2", "Store_3", "Store_4", "Store_5"]:
            return jsonify({"error": f"Invalid store_id: {store_id}"}), 400
        if category not in ["Electronics", "Apparel", "Grocery", "Home_Kitchen", "Beauty_Personal_Care"]:
            return jsonify({"error": f"Invalid category: {category}"}), 400
        if weather_event not in ["Clear", "Rain", "Snow"]:
            return jsonify({"error": f"Invalid weather_event: {weather_event}"}), 400
            
        # Run Forecast
        results, summary = run_recursive_forecast(
            start_date=start_date,
            end_date=end_date,
            store_id=store_id,
            category=category,
            promotion_active=promotion_active,
            competitor_discount=competitor_discount,
            weather_event=weather_event,
            is_stockout=is_stockout
        )
        
        return jsonify({
            "status": "success",
            "summary": summary,
            "data": results
        })
        
    except Exception as e:
        app.logger.error(f"Prediction Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run server locally on port 5000
    app.run(host="127.0.0.1", port=5000, debug=True)
