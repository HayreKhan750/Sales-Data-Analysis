import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Superstore Sales Dashboard")

# Load Data
df = pd.read_csv('data/superstore.csv')
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

# Filter by Region
regions = df['Region'].unique()
region = st.selectbox("Select Region", regions)
filtered = df[df['Region'] == region]

st.write(f"Showing data for: {region}")

# Total Sales
total_sales = filtered['Sales'].sum()
st.metric("Total Sales", f"${total_sales:,.2f}")

# Monthly Sales
monthly = filtered.groupby(filtered['Order Date'].dt.to_period('M'))['Sales'].sum()

fig, ax = plt.subplots()
monthly.plot(ax=ax)
ax.set_title('Monthly Sales Trend')
ax.set_xlabel('Month')
ax.set_ylabel('Sales')
st.pyplot(fig)
