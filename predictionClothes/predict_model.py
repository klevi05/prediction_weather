import sys
import pandas as pd
import joblib

sys.path.append('/mnt/c/Users/klevi/Desktop/flaskSoftware')  # Add the correct parent directory
from predictionTirana.predict_model import generate_forecast_v2  # Import the function

def predict_outfits(future_weather_dict):
    """
    Predict outfits based on future weather data.

    Args:
        future_weather_dict (dict): Weather data dictionary with keys like 'temp_max', 'temp_min', etc.

    Returns:1
        pd.DataFrame: DataFrame containing the date and predicted outfit components.
    """
    # Load trained models and encoders
    model_dir = '/mnt/c/Users/klevi/Desktop/flaskSoftware/predictionClothes/'
    shoe_model = joblib.load(f'{model_dir}shoe_model.pkl')
    lower_body_model = joblib.load(f'{model_dir}lower_body_model.pkl')
    upper_body_model = joblib.load(f'{model_dir}upper_body_model.pkl')
    label_encoders = joblib.load(f'{model_dir}label_encoders.pkl')

    future_weather = pd.DataFrame(future_weather_dict)
    # Extract only the columns used for model prediction
    features = ['temp_max', 'temp_min', 'uv_index_max', 'rain_sum']
    future_weather_features = future_weather[features]

    # Predict
    future_predictions_shoes = shoe_model.predict(future_weather_features)
    future_predictions_lower_body = lower_body_model.predict(future_weather_features)
    future_predictions_upper_body = upper_body_model.predict(future_weather_features)
    # Decode predictions
    future_outfits = pd.DataFrame({
        'date': pd.date_range(start=pd.Timestamp.today().normalize(), periods=len(future_weather), freq='D'),
        'shoes': label_encoders['shoes'].inverse_transform(future_predictions_shoes),
        'lower_body': label_encoders['lower_body'].inverse_transform(future_predictions_lower_body),
        'upper_body': label_encoders['upper_body'].inverse_transform(future_predictions_upper_body)
    })
    result_dict = future_outfits.to_dict(orient='list')
    return result_dict

# Example usage
if __name__ == "__main__":
    # Replace with a custom weather dictionary or generate using the forecast function
    future_weather_dict = generate_forecast_v2()
    outfits = predict_outfits(future_weather_dict)
    print(outfits)
