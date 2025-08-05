import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# App setup
st.set_page_config(page_title="MindScope - Mental Health Risk Forecast", layout="wide")
st.title("🧠 MindScope: Youth Mental Health Risk Index Dashboard")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("GHSH_Pooled_Data1.csv")

df = load_data()

# Select country
countries = sorted(df["Country"].unique())
selected_country = st.selectbox("🌍 Select a country", countries)

# Filter by selected country
country_df = df[df["Country"] == selected_country]

# Define behaviour columns and weights
weights = {
    "Suicide_Attempt": 5,
    "No_Close_Friends": 3,
    "Smoke_Cig": 2,
    "Drink_Alcohol": 2,
    "Bullied": 4,
    "Lonely": 3,
    "Marijuana_Use": 2
}

# Auto-detect columns that exist in the dataset
available_behaviours = [col for col in weights if col in df.columns]
if not available_behaviours:
    st.error("No matching behaviour columns found in dataset.")
    st.stop()

# Compute risk index per year
country_df["Risk_Index"] = country_df[available_behaviours].apply(
    lambda row: sum(row[beh] * weights[beh] for beh in available_behaviours), axis=1
)

# Group by year and average
yearly_risk = country_df.groupby("Year")["Risk_Index"].mean().reset_index()

# Forecast next 3 years
X = yearly_risk[["Year"]]
y = yearly_risk["Risk_Index"]
model = LinearRegression()
model.fit(X, y)

last_year = X["Year"].max()
future_years = np.arange(last_year + 1, last_year + 4).reshape(-1, 1)
future_preds = model.predict(future_years)

# Combine actual + forecasted
all_years = np.concatenate([X.values.flatten(), future_years.flatten()])
all_risks = np.concatenate([y, future_preds])

# 📊 Plot the Risk Index
st.subheader(f"📈 Mental Health Risk Trend: {selected_country}")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(all_years, all_risks, marker='o', linestyle='-', color='crimson', label="Risk Index")
ax.axvline(x=last_year, color='gray', linestyle='--', label="Forecast Starts")
ax.set_xlabel("Year")
ax.set_ylabel("Mental Health Risk Index")
ax.set_title(f"Forecasting Risk Index for {selected_country}")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# Breakdown by behaviour
st.subheader("🧮 Behaviour Contribution Breakdown")
latest_year = country_df["Year"].max()
latest_data = country_df[country_df["Year"] == latest_year][available_behaviours].mean()

# Horizontal bar chart
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.barh(latest_data.index, [latest_data[b] * weights[b] for b in latest_data.index], color='salmon')
ax2.set_title(f"Risk Factor Contribution - {selected_country} ({latest_year})")
ax2.set_xlabel("Weighted Contribution to Risk Index")
st.pyplot(fig2)

# Show raw values
st.write("📊 Raw Behaviour Averages:", latest_data.round(2))
