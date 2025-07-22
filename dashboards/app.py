# import streamlit as st
# import pandas as pd
# import plotly.express as px

# # ---- Page Config ----
# st.set_page_config(
#     page_title="Superstore Dashboard 💼",
#     page_icon="📊",
#     layout="wide"
# )

# # ---- Custom CSS ----
# st.markdown("""
#     <style>
#     .main {
#         background-color: #f0f2f6;  /* Slightly softer background */
#         color: #1c1e21;  /* Dark text for readability */
#     }
#     .big-font {
#         font-size:22px !important;
#         font-weight: 600;
#         color: #0d3b66;
#     }
#     .metric {
#         font-size:24px !important;
#         font-weight: 700;
#         color: #2a9d8f;
#     }
#     /* Style for raw data table */
#     div[data-testid="stDataFrame"] {
#         background-color: #ffffff !important;
#         color: #212529 !important;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # ---- Title ----
# st.title("💼 Superstore Sales Analytics Dashboard")

# # ---- Load Data ----
# @st.cache_data
# def load_data():
#     df = pd.read_csv('data/superstore.csv')
#     df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
#     return df

# df = load_data()

# # ---- Sidebar ----
# st.sidebar.header("📌 Filters")

# # Region filter
# regions = df['Region'].dropna().unique().tolist()
# region = st.sidebar.selectbox("Select Region", ["All"] + regions)

# # Date filter
# min_date = df['Order Date'].min()
# max_date = df['Order Date'].max()
# date_range = st.sidebar.date_input(
#     "Select Date Range",
#     [min_date, max_date],
#     min_value=min_date,
#     max_value=max_date
# )

# # Category filter
# categories = df['Category'].dropna().unique().tolist()
# category = st.sidebar.multiselect("Select Categories", categories, default=categories)

# # ---- Apply Filters ----
# filtered_df = df.copy()

# if region != "All":
#     filtered_df = filtered_df[filtered_df['Region'] == region]

# filtered_df = filtered_df[
#     (filtered_df['Order Date'] >= pd.to_datetime(date_range[0])) &
#     (filtered_df['Order Date'] <= pd.to_datetime(date_range[1]))
# ]

# filtered_df = filtered_df[filtered_df['Category'].isin(category)]

# # ---- KPI ----
# total_sales = filtered_df['Sales'].sum()
# total_profit = filtered_df['Profit'].sum() if 'Profit' in filtered_df.columns else 0
# total_orders = filtered_df['Order ID'].nunique()
# total_customers = filtered_df['Customer ID'].nunique()

# # ---- KPIs ----
# col1, col2, col3, col4 = st.columns(4)

# col1.metric("💰 Total Sales", f"${total_sales:,.2f}")
# col2.metric("📈 Total Profit", f"${total_profit:,.2f}")
# col3.metric("🧾 Total Orders", f"{total_orders:,}")
# col4.metric("👥 Unique Customers", f"{total_customers:,}")

# # ---- Monthly Sales Trend ----
# monthly_sales = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
# monthly_sales['Order Date'] = monthly_sales['Order Date'].dt.to_timestamp()

# fig_line = px.line(
#     monthly_sales,
#     x='Order Date',
#     y='Sales',
#     markers=True,
#     title="📅 Monthly Sales Trend",
#     template='plotly_white',
#     line_shape='spline',
#     color_discrete_sequence=['#ef476f']  # Vibrant red-pink
# )
# fig_line.update_traces(line=dict(width=4))
# fig_line.update_layout(
#     title_font=dict(size=20, family='Arial Black', color='#073b4c'),
#     xaxis_title='Date',
#     yaxis_title='Sales ($)',
#     plot_bgcolor='rgba(0,0,0,0)',
#     paper_bgcolor='rgba(0,0,0,0)',
# )

# st.plotly_chart(fig_line, use_container_width=True)

# # ---- Sales by Category Pie ----
# category_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
# fig_pie = px.pie(
#     category_sales,
#     names='Category',
#     values='Sales',
#     title="📊 Sales by Category",
#     color_discrete_sequence=px.colors.qualitative.Plotly  # <-- fixed here
# )

# fig_pie.update_layout(
#     title_font=dict(size=20, family='Arial Black', color='#073b4c')
# )
# st.plotly_chart(fig_pie, use_container_width=True)

# # ---- Top 10 Products ----
# top_products = (
#     filtered_df.groupby('Product Name')['Sales']
#     .sum()
#     .sort_values(ascending=False)
#     .head(10)
#     .reset_index()
# )

# fig_bar = px.bar(
#     top_products,
#     x='Sales',
#     y='Product Name',
#     orientation='h',
#     title="🏆 Top 10 Products",
#     color='Sales',
#     color_continuous_scale=px.colors.sequential.Agsunset
# )
# fig_bar.update_layout(
#     title_font=dict(size=20, family='Arial Black', color='#073b4c'),
#     yaxis=dict(autorange="reversed"),  # Reverse y-axis for descending order
#     plot_bgcolor='rgba(0,0,0,0)',
#     paper_bgcolor='rgba(0,0,0,0)',
# )
# fig_bar.update_traces(marker_line_color='black', marker_line_width=1.5)

# st.plotly_chart(fig_bar, use_container_width=True)

# # ---- Customer Segmentation ----
# st.subheader("📌 Customer Segment Overview")

# if 'Profit' in filtered_df.columns:
#     segment_df = filtered_df.groupby('Segment').agg(
#         Orders=('Order ID', 'nunique'),
#         Sales=('Sales', 'sum'),
#         Profit=('Profit', 'sum')
#     ).reset_index()
# else:
#     segment_df = filtered_df.groupby('Segment').agg(
#         Orders=('Order ID', 'nunique'),
#         Sales=('Sales', 'sum')
#     ).reset_index()
#     segment_df['Profit'] = 0

# fig_segment = px.bar(
#     segment_df,
#     x='Segment',
#     y='Sales',
#     color='Segment',
#     text_auto='.2s',
#     title="👥 Sales by Segment",
#     template='plotly_white',
#     color_discrete_sequence=px.colors.qualitative.Bold
# )
# fig_segment.update_layout(
#     title_font=dict(size=20, family='Arial Black', color='#073b4c'),
#     plot_bgcolor='rgba(0,0,0,0)',
#     paper_bgcolor='rgba(0,0,0,0)'
# )
# st.plotly_chart(fig_segment, use_container_width=True)

# # ---- Download Button ----
# st.download_button(
#     label="⬇️ Download Filtered Data as CSV",
#     data=filtered_df.to_csv(index=False).encode('utf-8'),
#     file_name="filtered_superstore.csv",
#     mime='text/csv'
# )

# # ---- Raw Data ----
# with st.expander("🔍 Show Raw Data"):
#     st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

# # ---- Footer ----
# st.markdown("---")
# st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io/) & [Plotly](https://plotly.com/python/)")













import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ---- Page Config ----
st.set_page_config(
    page_title="Superstore Dashboard & ML Model",
    page_icon="📊",
    layout="wide"
)

# ---- Page Selector in Sidebar ----
page = st.sidebar.radio("Select Page", ["Dashboard", "ML Prediction"])

# ---- DASHBOARD PAGE (EXACTLY YOUR CODE, UNCHANGED) ----
if page == "Dashboard":
    # ---- Custom CSS ----
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
            color: #1c1e21;
        }
        .big-font {
            font-size:22px !important;
            font-weight: 600;
            color: #0d3b66;
        }
        .metric {
            font-size:24px !important;
            font-weight: 700;
            color: #2a9d8f;
        }
        div[data-testid="stDataFrame"] {
            background-color: #ffffff !important;
            color: #212529 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("💼 Superstore Sales Analytics Dashboard")

    @st.cache_data
    def load_data():
        df = pd.read_csv('data/superstore.csv')
        df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
        return df

    df = load_data()

    st.sidebar.header("📌 Filters")

    regions = df['Region'].dropna().unique().tolist()
    region = st.sidebar.selectbox("Select Region", ["All"] + regions)

    min_date = df['Order Date'].min()
    max_date = df['Order Date'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    categories = df['Category'].dropna().unique().tolist()
    category = st.sidebar.multiselect("Select Categories", categories, default=categories)

    filtered_df = df.copy()
    if region != "All":
        filtered_df = filtered_df[filtered_df['Region'] == region]
    filtered_df = filtered_df[
        (filtered_df['Order Date'] >= pd.to_datetime(date_range[0])) &
        (filtered_df['Order Date'] <= pd.to_datetime(date_range[1]))
    ]
    filtered_df = filtered_df[filtered_df['Category'].isin(category)]

    total_sales = filtered_df['Sales'].sum()
    total_profit = filtered_df['Profit'].sum() if 'Profit' in filtered_df.columns else 0
    total_orders = filtered_df['Order ID'].nunique()
    total_customers = filtered_df['Customer ID'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Sales", f"${total_sales:,.2f}")
    col2.metric("📈 Total Profit", f"${total_profit:,.2f}")
    col3.metric("🧾 Total Orders", f"{total_orders:,}")
    col4.metric("👥 Unique Customers", f"{total_customers:,}")

    monthly_sales = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
    monthly_sales['Order Date'] = monthly_sales['Order Date'].dt.to_timestamp()
    fig_line = px.line(
        monthly_sales,
        x='Order Date',
        y='Sales',
        markers=True,
        title="📅 Monthly Sales Trend",
        template='plotly_white',
        line_shape='spline',
        color_discrete_sequence=['#ef476f']
    )
    fig_line.update_traces(line=dict(width=4))
    fig_line.update_layout(
        title_font=dict(size=20, family='Arial Black', color='#073b4c'),
        xaxis_title='Date',
        yaxis_title='Sales ($)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_line, use_container_width=True)

    category_sales = filtered_df.groupby('Category')['Sales'].sum().reset_index()
    fig_pie = px.pie(
        category_sales,
        names='Category',
        values='Sales',
        title="📊 Sales by Category",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig_pie.update_layout(title_font=dict(size=20, family='Arial Black', color='#073b4c'))
    st.plotly_chart(fig_pie, use_container_width=True)

    top_products = (
        filtered_df.groupby('Product Name')['Sales']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig_bar = px.bar(
        top_products,
        x='Sales',
        y='Product Name',
        orientation='h',
        title="🏆 Top 10 Products",
        color='Sales',
        color_continuous_scale=px.colors.sequential.Agsunset
    )
    fig_bar.update_layout(
        title_font=dict(size=20, family='Arial Black', color='#073b4c'),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    fig_bar.update_traces(marker_line_color='black', marker_line_width=1.5)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("📌 Customer Segment Overview")
    if 'Profit' in filtered_df.columns:
        segment_df = filtered_df.groupby('Segment').agg(
            Orders=('Order ID', 'nunique'),
            Sales=('Sales', 'sum'),
            Profit=('Profit', 'sum')
        ).reset_index()
    else:
        segment_df = filtered_df.groupby('Segment').agg(
            Orders=('Order ID', 'nunique'),
            Sales=('Sales', 'sum')
        ).reset_index()
        segment_df['Profit'] = 0

    fig_segment = px.bar(
        segment_df,
        x='Segment',
        y='Sales',
        color='Segment',
        text_auto='.2s',
        title="👥 Sales by Segment",
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_segment.update_layout(
        title_font=dict(size=20, family='Arial Black', color='#073b4c'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_segment, use_container_width=True)

    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name="filtered_superstore.csv",
        mime='text/csv'
    )

    with st.expander("🔍 Show Raw Data"):
        st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

    st.markdown("---")
    st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io/) & [Plotly](https://plotly.com/python/)")

# ---- ML PREDICTION PAGE (NEW FEATURE) ----
elif page == "ML Prediction":
    st.title("🤖 Superstore Sales Prediction Model")

    # Load data again for prediction
    @st.cache_data
    def load_and_process_data():
        df = pd.read_csv('data/superstore.csv')
        df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')

        # Encode categorical features needed for model
        le_region = LabelEncoder()
        le_category = LabelEncoder()
        le_subcategory = LabelEncoder()
        le_segment = LabelEncoder()

        df = df.dropna(subset=['Sales', 'Region', 'Category', 'Sub-Category', 'Segment'])
        df['Region_enc'] = le_region.fit_transform(df['Region'])
        df['Category_enc'] = le_category.fit_transform(df['Category'])
        df['SubCategory_enc'] = le_subcategory.fit_transform(df['Sub-Category'])
        df['Segment_enc'] = le_segment.fit_transform(df['Segment'])

        return df, le_region, le_category, le_subcategory, le_segment

    df_ml, le_region, le_category, le_subcategory, le_segment = load_and_process_data()

    # Train model once
    @st.cache_resource
    def train_model():
        features = ['Region_enc', 'Category_enc', 'SubCategory_enc', 'Segment_enc']
        target = 'Sales'

        X = df_ml[features]
        y = df_ml[target]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    model = train_model()

    st.markdown("### Enter feature values to predict Sales:")

    region_input = st.selectbox("Region", options=le_region.classes_)
    category_input = st.selectbox("Category", options=le_category.classes_)
    subcategory_input = st.selectbox("Sub-Category", options=le_subcategory.classes_)
    segment_input = st.selectbox("Segment", options=le_segment.classes_)

    # Encode inputs for prediction
    input_df = pd.DataFrame({
        'Region_enc': [le_region.transform([region_input])[0]],
        'Category_enc': [le_category.transform([category_input])[0]],
        'SubCategory_enc': [le_subcategory.transform([subcategory_input])[0]],
        'Segment_enc': [le_segment.transform([segment_input])[0]],
    })

    if st.button("Predict Sales"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Sales: ${prediction:,.2f}")

    st.markdown("---")
    st.info("This page predicts sales based on your selected features using a trained Random Forest model.")

