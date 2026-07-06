import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure reproducibility
np.random.seed(42)

def generate_sales_data(start_date="2021-01-01", end_date="2025-12-31"):
    print("Generating scaled synthetic sales data (5 years, 5 stores, 5 categories)...")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    stores = ["Store_1", "Store_2", "Store_3", "Store_4", "Store_5"]
    categories = ["Electronics", "Apparel", "Grocery", "Home_Kitchen", "Beauty_Personal_Care"]
    
    store_multipliers = {
        "Store_1": 1.0,
        "Store_2": 1.25,
        "Store_3": 0.85,
        "Store_4": 1.10,
        "Store_5": 0.95
    }
    
    category_bases = {
        "Grocery": 500,
        "Apparel": 200,
        "Electronics": 80,
        "Home_Kitchen": 150,
        "Beauty_Personal_Care": 120
    }
    
    # Define US holidays for 2021-2025
    holidays = []
    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
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
    holiday_set = set(holidays)
    
    data_rows = []
    
    for date in dates:
        day_of_week = date.dayofweek
        day_of_year = date.dayofyear
        year = date.year
        month = date.month
        
        # 1. Temperature: seasonal
        temp_base = 65 + 20 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        temperature = temp_base + np.random.normal(0, 4)
        
        is_holiday = date in holiday_set
        
        # 2. Weather Event Generation (Seasonal)
        # Winter months: December, January, February
        if month in [12, 1, 2]:
            weather_prob = np.random.rand()
            if weather_prob < 0.12:
                weather = "Snow"
            elif weather_prob < 0.35:
                weather = "Rain"
            else:
                weather = "Clear"
        # Spring/Summer/Autumn
        else:
            weather_prob = np.random.rand()
            if weather_prob < 0.20:
                weather = "Rain"
            else:
                weather = "Clear"
                
        # 3. Competitor Discount Active (10% overall chance)
        comp_hash = hash(f"{date.strftime('%Y-%m-%d')}") % 100
        competitor_discount = 1 if comp_hash < 10 else 0
        
        for store in stores:
            for category in categories:
                # Promotion flag (12% promo days)
                promo_hash = hash(f"{date.strftime('%Y-%m-%d')}_{store}_{category}") % 100
                promotion_active = 1 if promo_hash < 12 else 0
                
                # Out of Stock (Stockout) flag (2% chance independently)
                stockout_hash = hash(f"{date.strftime('%Y-%m-%d')}_{store}_{category}_stock") % 100
                is_stockout = 1 if stockout_hash < 2 else 0
                
                # Base Sales
                base = category_bases[category] * store_multipliers[store]
                
                # Trend: 4.5% year-over-year growth starting 2021
                year_fraction = (date - pd.Timestamp(start_date)).days / 365.25
                trend = 1.0 + (0.045 * year_fraction)
                
                # Weekly Seasonality
                if category == "Grocery":
                    weekly_factor = 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_week - 4) / 7)
                elif category == "Apparel":
                    weekly_factor = 1.0 + 0.25 * np.sin(2 * np.pi * (day_of_week - 5) / 7)
                elif category == "Electronics":
                    weekly_factor = 1.0 + 0.30 * np.sin(2 * np.pi * (day_of_week - 5) / 7)
                elif category == "Home_Kitchen":
                    weekly_factor = 1.0 + 0.18 * np.sin(2 * np.pi * (day_of_week - 5) / 7)
                else: # Beauty_Personal_Care
                    weekly_factor = 1.0 + 0.10 * np.sin(2 * np.pi * (day_of_week - 4) / 7)
                
                # Yearly Seasonality
                if category == "Electronics":
                    yearly_factor = 1.0 + 0.40 * np.exp(-((day_of_year - 330) / 25) ** 2)
                elif category == "Apparel":
                    yearly_factor = 1.0 + 0.15 * np.exp(-((day_of_year - 120) / 30) ** 2) + 0.20 * np.exp(-((day_of_year - 340) / 30) ** 2)
                elif category == "Home_Kitchen":
                    # Peaks in early summer (wedding season day 160) and Nov/Dec
                    yearly_factor = 1.0 + 0.12 * np.exp(-((day_of_year - 160) / 30) ** 2) + 0.18 * np.exp(-((day_of_year - 335) / 25) ** 2)
                elif category == "Beauty_Personal_Care":
                    # Smooth year round, small bump in Dec holiday season
                    yearly_factor = 1.0 + 0.10 * np.exp(-((day_of_year - 345) / 20) ** 2)
                else: # Grocery
                    yearly_factor = 1.0 + 0.05 * np.sin(2 * np.pi * (day_of_year - 170) / 365)
                
                # Holiday effect
                holiday_factor = 1.0
                if is_holiday:
                    if category == "Grocery":
                        holiday_factor = 1.35
                    elif category == "Electronics":
                        holiday_factor = 1.55
                    elif category == "Apparel":
                        holiday_factor = 1.45
                    elif category == "Home_Kitchen":
                        holiday_factor = 1.30
                    else: # Beauty
                        holiday_factor = 1.25
                
                # Promotion effect
                promo_factor = 1.0
                if promotion_active:
                    if category == "Electronics":
                        promo_factor = 1.50
                    elif category == "Apparel":
                        promo_factor = 1.40
                    elif category == "Home_Kitchen":
                        promo_factor = 1.30
                    elif category == "Beauty_Personal_Care":
                        promo_factor = 1.25
                    else: # Grocery
                        promo_factor = 1.15
                        
                # Weather effect
                weather_factor = 1.0
                if weather == "Rain":
                    if category != "Grocery":
                        weather_factor = 0.94 # 6% drop
                elif weather == "Snow":
                    if category == "Grocery":
                        weather_factor = 0.92 # 8% drop
                    else:
                        weather_factor = 0.83 # 17% drop due to snowstorm
                        
                # Competitor discount factor (drops store sales if active)
                competitor_factor = 0.88 if competitor_discount else 1.0
                
                # Calculate sales
                sales = base * trend * weekly_factor * yearly_factor * holiday_factor * promo_factor * weather_factor * competitor_factor
                
                # Add Gaussian noise
                noise_sd = base * 0.07
                sales += np.random.normal(0, noise_sd)
                
                # Stockout event overrides sales to 0
                if is_stockout:
                    sales = 0
                else:
                    sales = max(0, round(sales))
                
                data_rows.append({
                    "Date": date,
                    "Store_ID": store,
                    "Product_Category": category,
                    "Temperature": round(temperature, 1),
                    "Weather_Event": weather,
                    "Is_Holiday": 1 if is_holiday else 0,
                    "Promotion_Active": promotion_active,
                    "Competitor_Discount_Active": competitor_discount,
                    "Is_Stockout": is_stockout,
                    "Sales": sales
                })
                
    df = pd.DataFrame(data_rows)
    print(f"Data generation complete. Total rows: {len(df)}")
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Generate 5 years
    df_sales = generate_sales_data()
    
    # Save
    output_path = os.path.join("data", "raw_sales.csv")
    df_sales.to_csv(output_path, index=False)
    print(f"Saved raw data to {output_path}")
    
    # Plot weekly average sales trend for Store_1 - Electronics
    plt.figure(figsize=(12, 6))
    sample_df = df_sales[(df_sales["Store_ID"] == "Store_1") & (df_sales["Product_Category"] == "Electronics")].copy()
    sample_df.set_index("Date", inplace=True)
    weekly_sample = sample_df["Sales"].resample("W").mean()
    
    plt.plot(weekly_sample.index, weekly_sample.values, label="Store_1 - Electronics (Weekly Average)", color="#e91e63", linewidth=2)
    plt.title("Weekly Average Electronics Sales (Store_1: 2021-2025)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Sales Units", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    plot_path = os.path.join("plots", "raw_sales_trend.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved updated trend plot to {plot_path}")
