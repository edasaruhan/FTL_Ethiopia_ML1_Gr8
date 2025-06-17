import pickle
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and scaler
with open('models/model_rf.pkl', 'rb') as f:
    model_rf = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    sc = joblib.load(f)

# Feature columns
feature_cols = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

API_KEY = "42f07f9873614afa9ca180923251206"

ethiopian_crops = ["teff", "coffee", "maize", "sorghum", "wheat", "barley"]

low_rainfall_msgs = {
    "teff": "Teff may suffer from poor germination or stunted growth. Consider supplemental irrigation.",
    "coffee": "Coffee plants may show leaf drop or reduced flowering. Use mulch to retain soil moisture.",
    "maize": "Maize yield may be reduced due to water stress. Consider early irrigation.",
    "sorghum": "Sorghum is drought-resistant but still needs sufficient moisture. Monitor soil conditions.",
    "wheat": "Wheat may not tiller properly under dry conditions. Irrigation may be necessary.",
    "barley": "Barley is sensitive to drought during early stages. Irrigate if possible."
}

excess_rainfall_msgs = {
    "teff": "Teff may lodge or rot due to waterlogging. Use raised beds or improve drainage.",
    "coffee": "Coffee is vulnerable to fungal diseases. Ensure good aeration and avoid wet feet.",
    "maize": "Maize may lodge or suffer root rot. Avoid overwatering and manage drainage.",
    "sorghum": "Sorghum tolerates dry better than wet. Ensure fields are well-drained.",
    "wheat": "Excess rain can delay wheat harvest and increase disease. Monitor field moisture.",
    "barley": "Barley dislikes excessive moisture. Risk of fungal infection is high."
}

def get_day_of_year(date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    return date.timetuple().tm_yday

def fetch_weather_data(city, country, date):
    location = f"{city},{country}"
    url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={location}&dt={date}"
    response = requests.get(url)
    data = response.json()
    forecast = data['forecast']['forecastday'][0]['day']
    hour_data = data['forecast']['forecastday'][0]['hour'][12]  # Noon

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
            value = temp - ((100 - hum) / 5)
        elif col == 'humidity':
            value = forecast.get('avghumidity', 80.0)
        elif col == 'cloud':
            value = hour_data.get('cloud', 50.0)
        elif col == 'sunshine':
            cloud = hour_data.get('cloud', 50.0)
            value = max(0, 12 * (1 - cloud / 100))
        elif col == 'winddirection':
            value = hour_data.get('wind_degree', 180.0)
        elif col == 'windspeed':
            value = hour_data.get('wind_kph', 10.0)
        feature_values.append(float(value))

    total_precip_mm = forecast.get('totalprecip_mm', 0.0)  # Real rainfall in mm
    return feature_values, total_precip_mm

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        city = data['city']
        country = data['country']
        date = data.get('date', (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'))

        features, total_precip_mm = fetch_weather_data(city, country, date)
        input_data = pd.DataFrame([features], columns=feature_cols)
        input_scaled = sc.transform(input_data)

        proba = model_rf.predict_proba(input_scaled)[0, 1]
        prediction = "Rain" if proba >= 0.5 else "No Rain"
        rain_percent = proba * 100

        advisories = []
        for crop in ethiopian_crops:
            if total_precip_mm < 1:
                msg = f"⚠️ Very low rainfall predicted. {low_rainfall_msgs[crop]}"
            elif total_precip_mm > 20:
                msg = f"🚨 Heavy rainfall expected. {excess_rainfall_msgs[crop]}"
            else:
                msg = f"✅ Rainfall conditions appear suitable for {crop}. Proceed with standard practices."
            advisories.append({"crop": crop, "advisory": msg})

        return jsonify({
            "city": city,
            "country": country,
            "date": date,
            "rainfall_probability": round(proba, 2),
            "prediction": prediction,
            "total_precip_mm": round(total_precip_mm, 2),
            "advisories": advisories
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
