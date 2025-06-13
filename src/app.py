import pickle
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)

# Load model and scaler
with open('models/model_rf.pkl', 'rb') as f:
    model_rf = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    sc = joblib.load('models/scaler.pkl') 

# Feature columns (MUST match training data)
feature_cols = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

# WeatherAPI key
API_KEY = "42f07f9873614afa9ca180923251206"  # Replace if needed

def get_day_of_year(date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    return date.timetuple().tm_yday

def fetch_weather_data(city, country, date):
    location = f"{city},{country}"
    url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={location}&dt={date}"
    response = requests.get(url)
    data = response.json()
    
    forecast = data['forecast']['forecastday'][0]['day']
    hour_data = data['forecast']['forecastday'][0]['hour'][12]  # Noon data
    
    # Map API data to model features (EXACTLY as in Colab)
    feature_values = []
    for col in feature_cols:
        if col == 'day':
            value = get_day_of_year(date)
        elif col == 'pressure':
            value = hour_data.get('pressure_mb', 1012.0)
        elif col == 'maxtemp':
            value = forecast.get('maxtemp_c', 25.0)
        elif col == 'temparature':
            value = forecast.get('avgtemp_c', (forecast['maxtemp_c'] + forecast['mintemp_c']) / 2)
        elif col == 'mintemp':
            value = forecast.get('mintemp_c', 20.0)
        elif col == 'dewpoint':
            temp = forecast.get('avgtemp_c', 22.0)
            hum = forecast.get('avghumidity', 80.0)
            value = temp - ((100 - hum) / 5)  # Dewpoint approximation
        elif col == 'humidity':
            value = forecast.get('avghumidity', 80.0)
        elif col == 'cloud':
            value = hour_data.get('cloud', 50.0)
        elif col == 'sunshine':
            cloud = hour_data.get('cloud', 50.0)
            value = max(0, 12 * (1 - cloud / 100))  # Sunshine estimation
        elif col == 'winddirection':
            value = hour_data.get('wind_degree', 180.0)
        elif col == 'windspeed':
            value = hour_data.get('wind_kph', 10.0)
        feature_values.append(float(value))
    
    return feature_values

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        city = data['city']
        country = data['country']
        date = data.get('date', (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'))
        
        # Get and scale features
        features = fetch_weather_data(city, country, date)
        input_data = pd.DataFrame([features], columns=feature_cols)
        input_scaled = sc.transform(input_data)
        
        # Predict
        proba = model_rf.predict_proba(input_scaled)[0, 1]
        prediction = "Rain" if proba >= 0.5 else "No Rain"
        
        return jsonify({
            "city": city,
            "country": country,
            "date": date,
            "rainfall_probability": round(proba, 4),
            "prediction": prediction
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)