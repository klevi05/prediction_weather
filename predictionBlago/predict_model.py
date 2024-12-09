import pandas as pd
import joblib

def generate_forecast(forecast_steps=30):
    """
    Generates a 30-day forecast using pre-trained SARIMA models for specified parameters.

    Args:
        forecast_steps (int): Number of days to forecast (default is 30).

    Returns:
        dict: A dictionary containing forecast data for 'temp_max', 'temp_min', 'uv_index_max', and 'rain_sum'.
    """
    columns = ['temp_max', 'temp_min', 'uv_index_max', 'rain_sum']
    model_dir = '/mnt/c/Users/klevi/Desktop/flaskSoftware/predictionBlago/'  # Update to your model directory
    forecast_results = {}
    forecast_dates = pd.date_range(start=pd.Timestamp.today().normalize(), periods=forecast_steps, freq='D')

    for column in columns:
        try:
            print(f"Loading SARIMA model for {column}...")
            model = joblib.load(f"{model_dir}{column}_sarima_model.pkl")  # Use full path
            forecast = model.forecast(steps=forecast_steps)
            forecast_results[column] = forecast
        except Exception as e:
            print(f"Error loading or forecasting for {column}: {e}")
            forecast_results[column] = [None] * forecast_steps

    # Combine forecasts into a DataFrame
    forecast_df = pd.DataFrame(forecast_results, index=forecast_dates)
    forecast_df.index.name = 'date'

    # Post-process to enforce constraints
    forecast_df['uv_index_max'] = forecast_df['uv_index_max'].clip(lower=0.1)
    forecast_df['rain_sum'] = forecast_df['rain_sum'].clip(lower=0.0)
    # Convert DataFrame to dictionary with lists
    result_dict = forecast_df.to_dict(orient='list')
    result_dict['date'] = [date.strftime("%Y-%m-%d") for date in forecast_df.index]
    return result_dict
# Example usage
if __name__ == "__main__":
    forecast_data = generate_forecast()
