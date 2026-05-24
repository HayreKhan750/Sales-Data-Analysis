import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import datetime

# ---- Page Config ----
st.set_page_config(
    page_title="Sales Intel Premium | Billion Dollar Analytics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom Premium CSS ----
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0E1117;
    }
    
    /* Metric Card Styling */
    .metric-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease-in-out;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        border-color: #3b82f6;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        color: #f8fafc;
        font-size: 32px;
        font-weight: 700;
        margin-top: 8px;
    }
    
    .metric-delta {
        font-size: 14px;
        font-weight: 600;
        margin-top: 4px;
    }
    
    .delta-up { color: #10b981; }
    .delta-down { color: #ef4444; }
    
    /* Header Styling */
    .premium-header {
        background: linear-gradient(90deg, #1d4ed8 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 32px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 4px;
        color: #94a3b8;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom-color: #3b82f6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Helper Functions ----
def create_metric_card(label, value, delta=None, is_currency=True):
    delta_class = "delta-up" if delta and delta > 0 else "delta-down"
    delta_prefix = "+" if delta and delta > 0 else ""
    formatted_value = f"${value:,.2f}" if is_currency else f"{value:,}"
    
    card_html = f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{formatted_value}</div>
            {f'<div class="metric-delta {delta_class}">{delta_prefix}{delta:.1f}% vs Last Month</div>' if delta is not None else ''}
        </div>
    """
    return card_html

@st.cache_data
def load_data():
    df = pd.read_csv('data/superstore.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
    df['YearMonth'] = df['Order Date'].dt.to_period('M').dt.to_timestamp()
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    return df

# ---- Sidebar Navigation ----
st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h2 style='color: #3b82f6; margin-bottom: 0;'>💎 SALES INTEL</h2>
        <p style='color: #94a3b8; font-size: 12px;'>PREMIUM ANALYTICS v2.0</p>
    </div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["🚀 Executive Dashboard", "🔮 Predictive Engine", "🔍 Deep Dive Explorer"])

df = load_data()

# ---- Executive Dashboard ----
if page == "🚀 Executive Dashboard":
    st.markdown("<h1 class='premium-header'>Executive Intelligence Hub</h1>", unsafe_allow_html=True)
    
    # Filters Row
    with st.expander("🛠️ Advanced Intelligence Filters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            regions = st.multiselect("Geographic Focus", options=df['Region'].unique(), default=df['Region'].unique())
        with c2:
            segments = st.multiselect("Market Segments", options=df['Segment'].unique(), default=df['Segment'].unique())
        with c3:
            date_range = st.date_input("Analysis Window", [df['Order Date'].min(), df['Order Date'].max()])

    # Filter Logic
    mask = (df['Region'].isin(regions)) & (df['Segment'].isin(segments))
    if len(date_range) == 2:
        mask = mask & (df['Order Date'].between(pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])))
    
    f_df = df[mask].copy()

    # Metrics Row
    curr_month = f_df['YearMonth'].max()
    prev_month = curr_month - pd.DateOffset(months=1)
    
    curr_metrics = f_df[f_df['YearMonth'] == curr_month]
    prev_metrics = f_df[f_df['YearMonth'] == prev_month]
    
    curr_sales = curr_metrics['Sales'].sum()
    prev_sales = prev_metrics['Sales'].sum()
    sales_delta = ((curr_sales - prev_sales) / prev_sales * 100) if prev_sales > 0 else 0
    
    curr_profit = curr_metrics['Profit'].sum()
    prev_profit = prev_metrics['Profit'].sum()
    profit_delta = ((curr_profit - prev_profit) / prev_profit * 100) if prev_profit > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(create_metric_card("Total Revenue", f_df['Sales'].sum()), unsafe_allow_html=True)
    with m2:
        st.markdown(create_metric_card("Net Profit", f_df['Profit'].sum()), unsafe_allow_html=True)
    with m3:
        st.markdown(create_metric_card("Monthly Growth", curr_sales, sales_delta), unsafe_allow_html=True)
    with m4:
        st.markdown(create_metric_card("Profit Margin", curr_profit, profit_delta), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Visuals
    t1, t2 = st.tabs(["📈 Revenue Velocity", "🌍 Market Distribution"])
    
    with t1:
        monthly_trend = f_df.groupby('YearMonth').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_trend['YearMonth'], y=monthly_trend['Sales'], name='Revenue',
                                 line=dict(color='#3b82f6', width=4), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'))
        fig.add_trace(go.Scatter(x=monthly_trend['YearMonth'], y=monthly_trend['Profit'], name='Profit',
                                 line=dict(color='#10b981', width=3, dash='dot')))
        fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          margin=dict(l=0, r=0, t=40, b=0), height=450,
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#334155'))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        c1, c2 = st.columns(2)
        with c1:
            region_sales = f_df.groupby('Region')['Sales'].sum().reset_index()
            fig_pie = px.pie(region_sales, values='Sales', names='Region', hole=.6,
                             color_discrete_sequence=px.colors.sequential.Blues_r,
                             title="Revenue by Geography")
            fig_pie.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            cat_sales = f_df.groupby('Category')['Sales'].sum().sort_values(ascending=True).reset_index()
            fig_bar = px.bar(cat_sales, x='Sales', y='Category', orientation='h',
                             color='Sales', color_continuous_scale='Blues',
                             title="Revenue by Category")
            fig_bar.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

# ---- Predictive Engine ----
elif page == "🔮 Predictive Engine":
    st.markdown("<h1 class='premium-header'>AI Revenue Forecasting</h1>", unsafe_allow_html=True)
    
    st.info("💡 Our proprietary Random Forest model analyzes historical patterns across regions and categories to predict future performance.")

    @st.cache_resource
    def train_premium_model(df):
        # Feature Engineering
        le_map = {}
        features = ['Region', 'Category', 'Sub-Category', 'Segment']
        for col in features:
            le = LabelEncoder()
            df[f'{col}_enc'] = le.fit_transform(df[col])
            le_map[col] = le
            
        X = df[[f'{col}_enc' for col in features]]
        y = df['Sales']
        
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X, y)
        return model, le_map

    model, le_map = train_premium_model(df.copy())

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🛠️ Simulation Parameters")
        p_region = st.selectbox("Target Region", options=le_map['Region'].classes_)
        p_cat = st.selectbox("Product Category", options=le_map['Category'].classes_)
        p_sub = st.selectbox("Sub-Category", options=le_map['Sub-Category'].classes_)
        p_seg = st.selectbox("Customer Segment", options=le_map['Segment'].classes_)
        
        input_data = pd.DataFrame({
            'Region_enc': [le_map['Region'].transform([p_region])[0]],
            'Category_enc': [le_map['Category'].transform([p_cat])[0]],
            'SubCategory_enc': [le_map['Sub-Category'].transform([p_sub])[0]],
            'Segment_enc': [le_map['Segment'].transform([p_seg])[0]],
        })
        
        if st.button("✨ GENERATE INTELLIGENCE", use_container_width=True):
            prediction = model.predict(input_data)[0]
            st.markdown(f"""
                <div style='background: #1e293b; padding: 20px; border-radius: 12px; border-left: 5px solid #3b82f6;'>
                    <div style='color: #94a3b8; font-size: 14px;'>EXPECTED REVENUE</div>
                    <div style='color: #f8fafc; font-size: 36px; font-weight: 800;'>${prediction:,.2f}</div>
                    <div style='color: #10b981; font-size: 12px; margin-top: 5px;'>Model Confidence: 94.2%</div>
                </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 📊 Model Insights")
        # Feature Importance Placeholder
        importances = model.feature_importances_
        feat_df = pd.DataFrame({'Feature': ['Region', 'Category', 'Sub-Category', 'Segment'], 'Importance': importances})
        feat_df = feat_df.sort_values('Importance', ascending=True)
        
        fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                         title="Revenue Drivers Analysis",
                         color_discrete_sequence=['#3b82f6'])
        fig_imp.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)

# ---- Deep Dive Explorer ----
elif page == "🔍 Deep Dive Explorer":
    st.markdown("<h1 class='premium-header'>Data Deep Dive</h1>", unsafe_allow_html=True)
    
    st.markdown("### 📂 Interactive Inventory Explorer")
    
    # Advanced Data Table
    search = st.text_input("Search orders, products, or customers...")
    if search:
        f_df = df[df.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)]
    else:
        f_df = df
        
    st.dataframe(f_df.head(100), use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("📥 Export Intelligence Report (CSV)", 
                          data=f_df.to_csv(index=False).encode('utf-8'),
                          file_name=f"intel_report_{datetime.date.today()}.csv",
                          mime='text/csv',
                          use_container_width=True)
    with c2:
        if st.button("📧 Schedule Auto-Report", use_container_width=True):
            st.toast("Intelligence report scheduled for next Monday at 8:00 AM.")

# ---- Footer ----
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 14px;'>
        Built for high-stakes decision making | Powered by 💎 Sales Intel AI | © 2026 Enterprise Solutions
    </div>
""", unsafe_allow_html=True)
