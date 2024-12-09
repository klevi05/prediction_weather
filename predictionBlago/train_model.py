import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib  # To save the trained models

# Load the dataset
file_path = './data.csv'
data = pd.read_csv(file_path)

# Convert 'date' column to datetime and set it as the index
data['date'] = pd.to_datetime(data['date'], errors='coerce')
data = data.dropna(subset=['date'])  # Drop rows where the date could not be parsed
data.set_index('date', inplace=True)

# Ensure frequency information is set for the time index
data.index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')

# Ensure all columns are numeric
data = data.astype(float)

# Train SARIMA models for each column
models = {}
for column in data.columns:
    print(f"Training SARIMA model for {column}...")
    sarima_model = SARIMAX(data[column], order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    sarima_fit = sarima_model.fit(disp=False)
    models[column] = sarima_fit
    joblib.dump(sarima_fit, f"{column}_sarima_model.pkl")  # Save the model

print("All models trained and saved.")
