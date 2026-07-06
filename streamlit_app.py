import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# Set page config for wide premium layout
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme styling for clean premium look
st.markdown("""
<style>
    .main {
        background-color: #0c0f1d;
        color: #f3f4f6;
    }
    .stMetric {
        background-color: rgba(20, 24, 46, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px 20px;
    }
</style>
""", unsafe_allow_stdio=True)

# Path helpers - relative to project root (works locally and on Streamlit Cloud)
DATA_PATH = "data/raw_sales.csv"
MODEL_DIR = "models"

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

@st.cache_resource
def load_ml_artifacts():
    model_path = os.path.join(MODEL_DIR, "sales_forecast_model.joblib")
    features_path = os.path.join(MODEL_DIR, "feature_names.joblib")
    encoder_path = os.path.join(MODEL_DIR, "cat_encoder.joblib")
    
    if not all(os.path.exists(p) for p in [model_path, features_path, encoder_path]):
        return None, None, None
        
    model = joblib.load(model_path)
    feature_cols = joblib.load(features_path)
    encoder = joblib.load(encoder_path)
    return model, feature_cols, encoder

def run_recursive_forecast(start_date, end_date, store_id, category, promotion_active, competitor_discount, weather_event, is_stockout):
    model, feature_cols, encoder = load_ml_artifacts()
    
    if model is None:
        st.error("⚠️ Trained ML model files not found in the `models/` directory. Make sure you run `train.py` first.")
        return None
        
    if not os.path.exists(DATA_PATH):
        st.error(f"⚠️ Historical sales dataset not found at `{DATA_PATH}`.")
        return None
        
    raw_df = pd.read_csv(DATA_PATH)
    raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    
    # Filter historical values for seeding
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
    
    # Load history to compute lags/rolling window statistics
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
        
        # Temperature (seasonal simulation)
        temp_base = 65 + 20 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        temperature = round(temp_base + np.random.normal(0, 1.5), 1)
        
        is_holiday = 1 if date in holiday_list else 0
        days_since, days_to = calculate_single_holiday_distances(date, holiday_list)
        
        # Lags
        lag_7 = get_sales_on_date(date - pd.Timedelta(days=7))
        lag_14 = get_sales_on_date(date - pd.Timedelta(days=14))
        lag_30 = get_sales_on_date(date - pd.Timedelta(days=30))
        
        # Rolling window stats (shifted by 1 day)
        roll_7_vals = [get_sales_on_date(date - pd.Timedelta(days=i)) for i in range(1, 8)]
        roll_30_vals = [get_sales_on_date(date - pd.Timedelta(days=i)) for i in range(1, 31)]
        
        roll_mean_7 = np.mean(roll_7_vals)
        roll_mean_30 = np.mean(roll_30_vals)
        roll_std_7 = np.std(roll_7_vals)
        roll_std_30 = np.std(roll_30_vals)
        
        # One-hot encoding categories
        cat_df = pd.DataFrame([[store_id, category, weather_event]], columns=["Store_ID", "Product_Category", "Weather_Event"])
        encoded_cats = encoder.transform(cat_df)
        encoded_cols = encoder.get_feature_names_out(["Store_ID", "Product_Category", "Weather_Event"])
        encoded_dict = dict(zip(encoded_cols, encoded_cats[0]))
        
        # Build features payload
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
        
    # Extract results within user date boundary
    results = []
    for r in history_records:
        if target_start <= r["Date"] <= target_end:
            d_val = r["Date"]
            is_hol = "Yes" if d_val in holiday_list else "No"
            temp = round(65 + 20 * np.sin(2 * np.pi * (d_val.dayofyear - 100) / 365), 1)
            results.append({
                "Date": d_val.strftime("%Y-%m-%d"),
                "Day": d_val.strftime("%A"),
                "Weather": weather_event,
                "Holiday": is_hol,
                "Temperature (°F)": temp,
                "Forecasted Sales": int(r["Sales"])
            })
            
    return pd.DataFrame(results)

# ----------------- UI Rendering -----------------

st.title("📈 Sales Forecast Prediction Dashboard")
st.caption("AI-Powered Live Autoregressive Forecast Model using XGBoost")

# Sidebar Configuration panel
st.sidebar.header("🔧 Model Configuration")

start_date = st.sidebar.date_input("Start Date", pd.Timestamp("2026-01-01"))
end_date = st.sidebar.date_input("End Date", pd.Timestamp("2026-01-14"))

store = st.sidebar.selectbox(
    "Target Store",
    ["Store_1", "Store_2", "Store_3", "Store_4", "Store_5"],
    format_func=lambda x: f"Store {x.split('_')[1]}"
)

category = st.sidebar.selectbox(
    "Product Category",
    ["Electronics", "Apparel", "Grocery", "Home_Kitchen", "Beauty_Personal_Care"],
    format_func=lambda x: x.replace('_', ' ')
)

weather = st.sidebar.selectbox(
    "Weather Scenario",
    ["Clear", "Rain", "Snow"],
    help="Rain reduces non-grocery sales by 6%. Snow storms trigger a 17% retail traffic collapse."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Marketing & Operations")

promo = st.sidebar.checkbox("Active Promotional Campaign", value=False)
comp_promo = st.sidebar.checkbox("Competitor Promotions Active", value=False)
stockout = st.sidebar.checkbox("Force Stockout (No Inventory)", value=False)

if start_date > end_date:
    st.error("Error: Start date must be before or equal to End date.")
else:
    # Run predictions
    with st.spinner("Generating recursive sales predictions..."):
        df_forecast = run_recursive_forecast(
            start_date=start_date,
            end_date=end_date,
            store_id=store,
            category=category,
            promotion_active=1 if promo else 0,
            competitor_discount=1 if comp_promo else 0,
            weather_event=weather,
            is_stockout=1 if stockout else 0
        )
        
    if df_forecast is not None and not df_forecast.empty:
        # 1. Metric display row
        col1, col2, col3 = st.columns(3)
        
        total_vol = df_forecast["Forecasted Sales"].sum()
        peak_demand = df_forecast["Forecasted Sales"].max()
        avg_demand = df_forecast["Forecasted Sales"].mean()
        
        col1.metric("Total Forecasted Volume", f"{total_vol:,} units")
        col2.metric("Peak Daily Demand", f"{peak_demand:,} units")
        col3.metric("Average Daily Sales", f"{round(avg_demand, 1):,} units")
        
        st.markdown("### 📊 Daily Sales Trend")
        
        # 2. Render line chart
        chart_data = df_forecast.copy()
        chart_data.set_index("Date", inplace=True)
        st.line_chart(chart_data["Forecasted Sales"])
        
        st.markdown("### 📋 Daily Breakdown")
        
        # 3. Render table
        st.dataframe(
            df_forecast,
            column_config={
                "Forecasted Sales": st.column_config.NumberColumn(format="%d units"),
                "Temperature (°F)": st.column_config.NumberColumn(format="%.1f °F")
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.success("✅ Forecast successfully computed using optimal XGBoost hyperparameters.")
