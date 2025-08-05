import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Load dataset
file_path = '/content/GHSH_Pooled_Data1.csv'
df = pd.read_csv(file_path)

# Step 1: Show all countries
countries = sorted(df['Country'].unique())
print("Available Countries:\n")
for c in countries:
    print("-", c)

selected_country = input("\nType country name exactly as shown above: ")

if selected_country not in countries:
    print("❌ Country not found.")
else:
    # Step 2: Filter country
    country_df = df[df['Country'] == selected_country]

    # Step 3: List available years
    available_years = sorted(country_df['Year'].unique())
    print(f"\nAvailable Years for {selected_country}: {available_years}")

    try:
        selected_year = int(input("\nEnter year from above list (e.g., 2016): "))
        if selected_year not in available_years:
            print("❌ Year not found.")
        else:
            # Step 4: Filter selected year
            year_df = country_df[country_df['Year'] == selected_year]

            # Step 5: Get behaviour columns
            non_behaviour_cols = ['Country', 'Year', 'Age Group', 'Sex']
            behaviour_cols = [col for col in df.columns if col not in non_behaviour_cols]

            # 📊 PLOT 1: Bar chart for selected year
            behaviour_avg = year_df[behaviour_cols].mean()

            plt.figure(figsize=(12, 6))
            plt.barh(behaviour_avg.index, behaviour_avg.values, color='red')
            plt.xlabel("Average Value")
            plt.title(f"Behaviour Averages in {selected_country} ({selected_year})")
            plt.grid(axis='x')
            plt.tight_layout()
            plt.show()

            # 📈 PLOT 2: Prediction (line plot for each behaviour)
            print("\n📈 Predicting next 3 years for each behaviour...\n")
            yearly_avg = country_df.groupby('Year')[behaviour_cols].mean().reset_index()

            plt.figure(figsize=(14, 8))

            for behaviour in behaviour_cols:
                X = yearly_avg[['Year']]
                y = yearly_avg[behaviour]

                model = LinearRegression()
                model.fit(X, y)

                # Predict existing
                y_pred = model.predict(X)

                # Forecast next 3 years
                last_year = yearly_avg['Year'].max()
                future_years = np.arange(last_year + 1, last_year + 4).reshape(-1, 1)
                future_preds = model.predict(future_years)

                # Combine for full plot
                all_years = np.concatenate([X.values.flatten(), future_years.flatten()])
                all_values = np.concatenate([y_pred, future_preds])

                plt.plot(all_years, all_values, marker='o', label=behaviour)

            plt.title(f'Behaviour Forecast (Next 3 Years) - {selected_country}')
            plt.xlabel("Year")
            plt.ylabel("Average Value")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    except ValueError:
        print("❌ Invalid year input.")
