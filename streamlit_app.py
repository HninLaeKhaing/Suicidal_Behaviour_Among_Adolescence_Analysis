import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Page title
st.set_page_config(page_title="YouthWatch: Real-Time Suicidal Behavioural Risk Forecast For Adolescents ", layout="wide")
st.title("📊 YouthWatch: Real-Time Suicidal Behavioural Risk Forecast For Adolescents")
st.markdown("Analyze youth behaviour patterns by country and predict future trends.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("GHSH_Pooled_Data1.csv")

df = load_data()

# Detect country list
countries = sorted(df['Country'].unique())
selected_country = st.selectbox("🌍 Select a Country", countries)

# Filter country data
country_df = df[df['Country'] == selected_country]

# Show available years
available_years = sorted(country_df['Year'].unique())
selected_year = st.selectbox("📅 Select a Year", available_years)

# Get behaviour columns automatically
non_behaviour_cols = ['Country', 'Year', 'Age Group', 'Sex']
behaviour_cols = [col for col in df.columns if col not in non_behaviour_cols]

# --- Plot 1: Bar chart for selected year ---
st.subheader(f"📌 Average Case in {selected_country} - {selected_year}")
year_df = country_df[country_df['Year'] == selected_year]
behaviour_avg = year_df[behaviour_cols].mean()

fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.barh(behaviour_avg.index, behaviour_avg.values, color='skyblue')
ax1.set_xlabel("Average Value")
ax1.set_title(f"Behaviour Averages - {selected_country} ({selected_year})")
ax1.grid(axis='x')
st.pyplot(fig1)

# --- Plot 2: Forecasting next 3 years ---
st.subheader(f"📈 Forecasting Next 3 Years - {selected_country}")
yearly_avg = country_df.groupby('Year')[behaviour_cols].mean().reset_index()

fig2, ax2 = plt.subplots(figsize=(14, 8))

for behaviour in behaviour_cols:
    X = yearly_avg[['Year']]
    y = yearly_avg[behaviour]

    model = LinearRegression()
    model.fit(X, y)

    # Predict for existing and future
    y_pred = model.predict(X)
    last_year = yearly_avg['Year'].max()
    future_years = np.arange(last_year + 1, last_year + 4).reshape(-1, 1)
    future_preds = model.predict(future_years)

    # Combine all
    all_years = np.concatenate([X.values.flatten(), future_years.flatten()])
    all_values = np.concatenate([y_pred, future_preds])

    ax2.plot(all_years, all_values, marker='o', label=behaviour)

ax2.set_xlabel("Year")
ax2.set_ylabel("Average Value")
ax2.set_title(f"Forecast in {selected_country} (Next 3 Years)")
ax2.grid(True)
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
st.pyplot(fig2)
