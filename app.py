import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker 
import os 

st.set_page_config( 
    page_title="Sales Intelligence Dashboard", 
    page_icon="📊", 
    layout="wide" 
) 

st.markdown(""" 
<style> 
    .metric-box { 
        background: #1E2130; 
        border-radius: 10px; 
        padding: 16px 20px; 
        border: 1px solid #2E3250; 
        text-align: center; 
    } 
    .metric-val { font-size: 28px; font-weight: bold; color: #0077B5; } 
    .metric-lbl { font-size: 13px; color: #8899BB; margin-top: 4px; } 
</style> 
""", unsafe_allow_html=True) 

st.title("📊 Predictive Sales Intelligence Dashboard") 
st.markdown("**Python · Pandas · Matplotlib · Built by ` `https://hayredin.vercel.app` ` **") 
st.divider() 

# ── Load or generate data ───────────────────────────────────── 
@st.cache_data 
def load_data(): 
    # Try loading your actual CSV — update filename to match yours 
    for fname in ['sales.csv','data.csv','sales_data.csv','dataset.csv', 'data/superstore.csv']: 
        if os.path.exists(fname): 
            data = pd.read_csv(fname)
            # Standardize column names for superstore.csv if detected
            if 'Category' in data.columns and 'Product Name' in data.columns:
                data = data.rename(columns={
                    'Order Date': 'Date',
                    'Category': 'Product',
                    'Sales': 'Revenue'
                })
            return data

    # Fallback: generate realistic sample data 
    np.random.seed(42) 
    months  = pd.date_range('2023-01-01', periods=24, freq='ME') 
    regions = ['East Africa','West Africa','North Africa','South Africa'] 
    products= ['Software','Hardware','Services','Consulting'] 
    rows = [] 
    for month in months: 
        for region in regions: 
            for product in products: 
                base   = {'Software':45000,'Hardware':32000, 
                          'Services':28000,'Consulting':38000}[product] 
                season = 1.3 if month.month in [11,12] else (1.1 if month.month in [3,4] else 1.0)
                rev = base * season * np.random.uniform(0.8, 1.2) 
                rows.append({ 
                    'Date': month, 'Region': region, 
                    'Product': product, 'Revenue': round(rev, 2), 
                    'Units': int(rev / np.random.uniform(80,120)) 
                }) 
    return pd.DataFrame(rows) 

df = load_data() 

# ── Sidebar filters ─────────────────────────────────────────── 
st.sidebar.header("🔧 Filters") 
regions  = st.sidebar.multiselect("Region",  df['Region'].unique(), 
                                   default=df['Region'].unique()) 
products = st.sidebar.multiselect("Product", df['Product'].unique(), 
                                   default=df['Product'].unique()) 
filtered = df[df['Region'].isin(regions) & df['Product'].isin(products)] 

# ── KPI row ─────────────────────────────────────────────────── 
total_rev  = filtered['Revenue'].sum() 
avg_rev    = filtered.groupby('Date')['Revenue'].sum().mean() 
top_region = filtered.groupby('Region')['Revenue'].sum().idxmax() 
top_product= filtered.groupby('Product')['Revenue'].sum().idxmax() 

c1,c2,c3,c4 = st.columns(4) 
with c1: 
    st.markdown(f'<div class="metric-box"><div class="metric-val">${total_rev:,.0f}</div><div class="metric-lbl">Total Revenue</div></div>', unsafe_allow_html=True) 
with c2: 
    st.markdown(f'<div class="metric-box"><div class="metric-val">${avg_rev:,.0f}</div><div class="metric-lbl">Avg Monthly Revenue</div></div>', unsafe_allow_html=True) 
with c3: 
    st.markdown(f'<div class="metric-box"><div class="metric-val">{top_region}</div><div class="metric-lbl">Top Region</div></div>', unsafe_allow_html=True) 
with c4: 
    st.markdown(f'<div class="metric-box"><div class="metric-val">{top_product}</div><div class="metric-lbl">Top Product</div></div>', unsafe_allow_html=True) 

st.divider() 

# ── Charts ──────────────────────────────────────────────────── 
col1, col2 = st.columns(2) 

with col1: 
    st.subheader("📈 Monthly Revenue Trend") 
    monthly = filtered.groupby('Date')['Revenue'].sum().reset_index() 
    fig, ax  = plt.subplots(figsize=(7,4)) 
    ax.plot(monthly['Date'], monthly['Revenue'], 
            color='#0077B5', linewidth=2.5, marker='o', markersize=5) 
    ax.fill_between(monthly['Date'], monthly['Revenue'], 
                    alpha=0.15, color='#0077B5') 
    ax.set_facecolor('#1E2130') 
    fig.patch.set_facecolor('#1E2130') 
    ax.tick_params(colors='white') 
    ax.yaxis.set_major_formatter(mticker.FuncFormatter( 
        lambda x,_: f"${x/1000:.0f}k")) 
    ax.spines[:].set_color('#2E3250') 
    plt.xticks(rotation=45) 
    st.pyplot(fig) 

with col2: 
    st.subheader("🌍 Revenue by Region") 
    by_region = filtered.groupby('Region')['Revenue'].sum().sort_values() 
    fig, ax   = plt.subplots(figsize=(7,4)) 
    bars = ax.barh(by_region.index, by_region.values, 
                   color=['#0077B5','#00BFFF','#4DA6E0','#2E6DA4']) 
    ax.set_facecolor('#1E2130') 
    fig.patch.set_facecolor('#1E2130') 
    ax.tick_params(colors='white') 
    ax.xaxis.set_major_formatter(mticker.FuncFormatter( 
        lambda x,_: f"${x/1000:.0f}k")) 
    ax.spines[:].set_color('#2E3250') 
    for bar in bars: 
        ax.text(bar.get_width()+500, bar.get_y()+bar.get_height()/2, 
                f"${bar.get_width()/1000:.0f}k", 
                va='center', color='white', fontsize=10) 
    st.pyplot(fig) 

col3, col4 = st.columns(2) 

with col3: 
    st.subheader("📦 Revenue by Product") 
    by_product = filtered.groupby('Product')['Revenue'].sum() 
    fig, ax    = plt.subplots(figsize=(6,4)) 
    colors     = ['#0077B5','#00BFFF','#4DA6E0','#2E6DA4'] 
    ax.pie(by_product.values, labels=by_product.index, 
           colors=colors, autopct='%1.1f%%', 
           textprops={'color':'white'}) 
    fig.patch.set_facecolor('#1E2130') 
    st.pyplot(fig) 

with col4: 
    st.subheader("📅 Seasonal Revenue Heatmap") 
    filtered['Month'] = pd.to_datetime(filtered['Date']).dt.month 
    filtered['Year']  = pd.to_datetime(filtered['Date']).dt.year 
    pivot = filtered.pivot_table( 
        values='Revenue', index='Year', columns='Month', aggfunc='sum') 
    fig, ax = plt.subplots(figsize=(7,4)) 
    im = ax.imshow(pivot.values, cmap='Blues', aspect='auto') 
    ax.set_xticks(range(12)) 
    ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun', 
                        'Jul','Aug','Sep','Oct','Nov','Dec'], 
                       color='white', fontsize=9) 
    ax.set_yticks(range(len(pivot.index))) 
    ax.set_yticklabels(pivot.index, color='white') 
    fig.patch.set_facecolor('#1E2130') 
    ax.set_facecolor('#1E2130') 
    plt.colorbar(im, ax=ax).ax.tick_params(colors='white') 
    st.pyplot(fig) 

st.divider() 
st.markdown("*Built by ` `https://hayredin.vercel.app` `  · github.com/HayreKhan750*") 
