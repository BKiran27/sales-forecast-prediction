"""
Sales Forecasting Production Web Dashboard
Dual-Mode: Standalone / Streamlit Community Cloud & Airflow / MLflow
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add both root, ui, and ui/utils directories to sys.path
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
ui_utils_dir = os.path.join(curr_dir, "utils")

for p in [root_dir, curr_dir, ui_utils_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from ui.utils.simple_model_loader import SimpleModelLoader
    from ui.utils.simple_predictor import SimplePredictor
except ImportError:
    try:
        from utils.simple_model_loader import SimpleModelLoader
        from utils.simple_predictor import SimplePredictor
    except ImportError:
        from simple_model_loader import SimpleModelLoader
        from simple_predictor import SimplePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Sales Forecast Inference",
    page_icon="🔮",
    layout="wide"
)

# Initialize Session State
if 'model_loader' not in st.session_state:
    loader = SimpleModelLoader()
    st.session_state.model_loader = loader
    st.session_state.predictor = SimplePredictor(loader)
    st.session_state.models_loaded = loader.loaded
    st.session_state.run_id = "pre_trained_local" if loader.loaded else None

# Default seed data on startup
if 'current_data' not in st.session_state:
    root_dir = os.path.dirname(curr_dir)
    sample_csv = os.path.join(root_dir, "data", "sample_sales_data.csv")
    if os.path.exists(sample_csv):
        try:
            st.session_state.current_data = pd.read_csv(sample_csv)
        except Exception:
            st.session_state.current_data = None
    else:
        st.session_state.current_data = None

if 'last_results' not in st.session_state:
    st.session_state.last_results = None

# Header
st.title("🔮 Enterprise Sales Forecast Inference")
st.markdown("Real-time end-to-end multi-model sales forecasting powered by **XGBoost**, **LightGBM**, and **Ensemble Modeling**.")

# Sidebar for Model Configuration & Settings
with st.sidebar:
    st.header("📦 Model Status")
    
    if not st.session_state.models_loaded:
        st.error("⚠️ No models currently loaded")
    else:
        st.success("✅ Models Loaded & Active")
        available_models = list(st.session_state.model_loader.models.keys())
        st.info(f"Available: **{', '.join([m.upper() for m in available_models])}**")
        if st.session_state.run_id:
            st.caption(f"Engine: {st.session_state.run_id}")
    
    if st.button("🔄 Reload Models", type="primary", use_container_width=True):
        with st.spinner("Checking local models & MLflow registry..."):
            loaded_local = st.session_state.model_loader.load_local_models()
            if loaded_local:
                st.session_state.models_loaded = True
                st.session_state.run_id = "pre_trained_local"
                st.success("✅ Models refreshed successfully!")
                st.rerun()
            else:
                run_id = st.session_state.model_loader.get_latest_run()
                if run_id and st.session_state.model_loader.load_models_from_run(run_id):
                    st.session_state.models_loaded = True
                    st.session_state.run_id = run_id
                    st.success("✅ Models loaded from MLflow!")
                    st.rerun()
                else:
                    st.error("❌ Failed to reload models")
    
    st.markdown("---")
    st.header("⚙️ Forecast Settings")
    
    model_type = st.selectbox(
        "Forecast Model",
        ["ensemble", "xgboost", "lightgbm"],
        index=0,
        help="Ensemble blends XGBoost and LightGBM to minimize variance"
    )
    
    forecast_days = st.slider(
        "Forecast Horizon (Days)",
        min_value=1,
        max_value=90,
        value=30,
        help="Number of days to forecast into the future"
    )

# Main Application Body
if st.session_state.models_loaded:
    tab1, tab2, tab3 = st.tabs(["🎲 Active / Sample Data", "📤 Upload Custom CSV", "✏️ Manual Daily Entry"])
    
    # TAB 1: Active or Pre-loaded Sample Data
    with tab1:
        st.markdown("### Pre-Loaded Sales Dataset")
        st.markdown("Use the built-in realistic multi-store sales data or generate a customized scenario:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sample_days = st.number_input("Days of History", value=60, min_value=7, max_value=365)
        with col2:
            avg_sales = st.number_input("Average Baseline Sales ($)", value=5000, min_value=100)
        with col3:
            volatility = st.slider("Demand Volatility (%)", 0, 50, 15)
            
        if st.button("⚡ Generate New Random Scenario", key="gen_sample_btn"):
            dates = pd.date_range(end=datetime.now(), periods=sample_days, freq='D')
            trend = np.linspace(0, avg_sales * 0.1, sample_days)
            seasonal = avg_sales * 0.2 * np.sin(2 * np.pi * np.arange(sample_days) / 7)
            noise = np.random.normal(0, avg_sales * volatility / 100, sample_days)
            sales = np.maximum(avg_sales + trend + seasonal + noise, 10.0)
            
            st.session_state.current_data = pd.DataFrame({
                'date': dates,
                'store_id': 'store_001',
                'sales': np.round(sales, 2)
            })
            st.session_state.last_results = None
            st.success("✅ Generated new scenario data!")
            st.rerun()

    # TAB 2: Upload CSV
    with tab2:
        st.markdown("### Upload Historical Sales Data (CSV)")
        uploaded_file = st.file_uploader(
            "Upload CSV file with columns: date, sales (and optionally store_id)",
            type=['csv'],
            help="Ensure date is formatted YYYY-MM-DD or standard datetime string."
        )
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                required_cols = ['date', 'sales']
                missing_cols = [c for c in required_cols if c not in uploaded_df.columns]
                if missing_cols:
                    st.error(f"❌ Uploaded CSV is missing required columns: {missing_cols}")
                else:
                    if 'store_id' not in uploaded_df.columns:
                        uploaded_df['store_id'] = 'store_001'
                    st.session_state.current_data = uploaded_df
                    st.session_state.last_results = None
                    st.success(f"✅ Successfully loaded {len(uploaded_df)} rows from CSV!")
            except Exception as e:
                st.error(f"❌ Failed to parse CSV: {e}")

    # TAB 3: Manual Daily Entry
    with tab3:
        st.markdown("### Quick 7-Day Manual Input")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            manual_store = st.text_input("Store Identifier", value="store_001")
        with col_s2:
            base_manual = st.number_input("Base Value ($)", value=4500, min_value=100)
            
        m_cols = st.columns(7)
        manual_records = []
        for i in range(7):
            d = datetime.now() - timedelta(days=6 - i)
            with m_cols[i]:
                st.caption(d.strftime('%a %m/%d'))
                val = st.number_input(
                    "Sales",
                    value=int(base_manual + (i * 120)),
                    key=f"m_val_{i}",
                    label_visibility="collapsed"
                )
                manual_records.append({
                    'date': d.strftime('%Y-%m-%d'),
                    'store_id': manual_store,
                    'sales': float(val)
                })
        if st.button("Apply Manual Data", key="apply_manual_btn"):
            st.session_state.current_data = pd.DataFrame(manual_records)
            st.session_state.last_results = None
            st.success("✅ Applied manual 7-day data!")
            st.rerun()

    # Active Dataset Review & Forecast Trigger
    input_data = st.session_state.current_data
    if input_data is not None and len(input_data) > 0:
        st.markdown("---")
        with st.expander("📋 Active Dataset Overview", expanded=False):
            c_info1, c_info2, c_info3 = st.columns(3)
            with c_info1:
                st.write(f"**Total Records**: {len(input_data)}")
            with c_info2:
                st.write(f"**Store ID**: {input_data['store_id'].iloc[0] if 'store_id' in input_data.columns else 'store_001'}")
            with c_info3:
                st.write(f"**Recent Mean Sales**: ${input_data['sales'].mean():,.2f}")
            st.dataframe(input_data.tail(10), use_container_width=True)

        # Centered Run Forecast Button
        _, col_btn, _ = st.columns([1, 2, 1])
        with col_btn:
            if st.button("🚀 Run Sales Forecast", type="primary", use_container_width=True, key="run_btn"):
                with st.spinner(f"Computing {forecast_days}-day forecast with {model_type.upper()}..."):
                    results = st.session_state.predictor.predict(
                        input_data,
                        model_type=model_type,
                        forecast_days=forecast_days
                    )
                    st.session_state.last_results = results

        # Display Forecast Results if available
        if st.session_state.last_results is not None:
            results = st.session_state.last_results
            if results.get('success'):
                st.success("✅ Forecast Generated Successfully!")
                
                # Metrics Row
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Total Projected Sales", f"${results['summary']['total_predicted_sales']:,.0f}")
                with m2:
                    st.metric("Projected Daily Average", f"${results['summary']['average_daily_sales']:,.0f}")
                with m3:
                    st.metric("Horizon", f"{forecast_days} Days")
                with m4:
                    st.metric("Model Architecture", results.get('model_type', model_type).upper())
                
                # Plotly Chart
                st.markdown("### 📈 Historical Sales & Future Forecast")
                predictions_df = results['predictions']
                
                fig = go.Figure()
                
                # 1. Historical Line
                hist_dates = pd.to_datetime(input_data['date'])
                fig.add_trace(go.Scatter(
                    x=hist_dates,
                    y=input_data['sales'],
                    mode='lines+markers',
                    name='Historical Sales',
                    line=dict(color='#1f77b4', width=2),
                    marker=dict(size=4)
                ))
                
                # 2. Predicted Line
                pred_dates = pd.to_datetime(predictions_df['date'])
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=predictions_df['predicted_sales'],
                    mode='lines+markers',
                    name=f'{model_type.upper()} Forecast',
                    line=dict(color='#2ca02c', width=3),
                    marker=dict(size=5)
                ))
                
                # 3. Upper Bound
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=predictions_df['upper_bound'],
                    fill=None,
                    mode='lines',
                    line=dict(color='rgba(0,0,0,0)', width=0),
                    showlegend=False
                ))
                
                # 4. Lower Bound with shaded region
                fig.add_trace(go.Scatter(
                    x=pred_dates,
                    y=predictions_df['lower_bound'],
                    fill='tonexty',
                    mode='lines',
                    line=dict(color='rgba(44, 160, 44, 0.2)', width=0),
                    fillcolor='rgba(44, 160, 44, 0.18)',
                    name='Confidence Interval'
                ))
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Sales ($)",
                    hovermode='x unified',
                    template='plotly_white',
                    height=520,
                    margin=dict(l=30, r=30, t=40, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Export and Data Table
                st.markdown("### 💾 Export & Review Forecast Data")
                export_cols = ['date', 'predicted_sales', 'lower_bound', 'upper_bound']
                export_df = predictions_df[export_cols].copy().round(2)
                
                col_exp1, col_exp2 = st.columns([1, 1])
                with col_exp1:
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast CSV",
                        data=csv_data,
                        file_name=f"sales_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        type="primary"
                    )
                with col_exp2:
                    st.caption("Forecast estimates include baseline prediction along with lower and upper confidence bounds.")
                    
                st.dataframe(export_df, use_container_width=True)
            else:
                st.error(f"❌ Prediction error: {results.get('error')}")
    else:
        st.info("👈 Please load or enter sales data using the tabs above to begin forecasting.")
else:
    st.warning("⚠️ Please load models from the sidebar before initiating forecasts.")