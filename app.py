from flask import Flask,request, jsonify
from pymongo import MongoClient
from flask_cors import CORS 
from flask_bcrypt import Bcrypt
import os
import sys
sys.path.append('/mnt/c/Users/klevi/Desktop/flaskSoftware') 
from predictionBlago.predict_model import generate_forecast
from predictionTirana.predict_model import generate_forecast_v2
from predictionClothes.predict_model import predict_outfits

app = Flask(__name__)
bcrypt = Bcrypt(app)
#connect to mongoDb
mongo_client = MongoClient(f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@prediction.30xch.mongodb.net/")  # Replace with your MongoDB URI
db = mongo_client['Prediction']  # Replace with your database name
collection = db['Users']
CORS(app, origins=["http://localhost:5173"])
# Define the user model
def validate_user(data):
    """
    Validate the user data before inserting into MongoDB.
    """
    required_fields = {'name', 'lastname', 'email', 'password'}
    if not all(field in data for field in required_fields):
        return False, "Missing required fields."
    return True, None

@app.route("/")
def home():
    return "hello"

@app.route("/add_user", methods=["POST"])
def add_user():
    """
    Add a new user to the MongoDB collection.
    Expects JSON payload with name, lastname, email, and password.
    """
    data = request.json
    is_valid, error_message = validate_user(data)
    if not is_valid:
        return jsonify({"error": error_message}), 400
    # Check if the email already exists
    existing_user = collection.find_one({"email": data.get("email")})
    if existing_user:
        return jsonify({"error": "Email already exists"}), 201

    # Insert the validated user into the MongoDB collection
    result = collection.insert_one(data)
    return jsonify({"message": "User added successfully", "id": str(result.inserted_id)})

@app.route("/blago", methods=['GET'])
async def blago_forecast():
    forecast_dict = generate_forecast()
    forecast_outfits = predict_outfits(forecast_dict)
    # Transform data into the required format
    formatted_data = [
        {
            "date": forecast_dict['date'][i],
            "rain_sum": forecast_dict["rain_sum"][i],
            "temp_max": forecast_dict["temp_max"][i],
            "temp_min": forecast_dict["temp_min"][i],
            "uv_index_max": forecast_dict["uv_index_max"][i],
            "lower_body": forecast_outfits["lower_body"][i],
            "shoes": forecast_outfits["shoes"][i],
            "upper_body": forecast_outfits["upper_body"][i]
        }
        for i in range(len(forecast_dict["rain_sum"]))
    ]
    # Return JSON response
    return jsonify(formatted_data)

@app.route("/tirana", methods=['GET'])
async def tirana_forecast():
    forecast_dict = generate_forecast_v2()
    forecast_outfits = predict_outfits(forecast_dict)
    # Transform data into the required format
    formatted_data = [
        {
            "date": forecast_dict['date'][i],
            "rain_sum": forecast_dict["rain_sum"][i],
            "temp_max": forecast_dict["temp_max"][i],
            "temp_min": forecast_dict["temp_min"][i],
            "uv_index_max": forecast_dict["uv_index_max"][i],
            "lower_body": forecast_outfits["lower_body"][i],
            "shoes": forecast_outfits["shoes"][i],
            "upper_body": forecast_outfits["upper_body"][i]
        }
        for i in range(len(forecast_dict["rain_sum"]))
    ]
    
    # Return JSON response
    return jsonify(formatted_data)

if __name__ == '__main__':
    app.run(debug=True) 