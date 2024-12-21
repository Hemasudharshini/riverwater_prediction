from flask import Flask, render_template, request
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Load the saved model and scaler
loaded_model = load_model('water_level_prediction_model.h5')
loaded_scaler = joblib.load('scaler_model.pkl')

def predict_water_level(input_data, scaler, model):
    # Ensure input_data is a DataFrame
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    
    # Normalize the input data
    input_scaled = scaler.transform(input_data)
    
    # Reshape for LSTM (sequence length of 1)
    input_scaled = input_scaled.reshape(input_scaled.shape[0], 1, input_scaled.shape[1])
    
    # Make prediction
    prediction = model.predict(input_scaled)
    return prediction[0][0]  # Return the predicted value

@app.route('/')
def base():
    return render_template('base.html')

@app.route('/predict-form')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        input_data = {
            'Historical_Water_Level': float(request.form['historical_water_level']),
            'Rainfall': float(request.form['rainfall']),
            'Temperature': float(request.form['temperature']),
            'Seasonality_Month': int(request.form['seasonality_month']),
            'Upstream_Flow': float(request.form['upstream_flow']),
            'Evaporation': float(request.form['evaporation']),
            'Soil_Moisture': float(request.form['soil_moisture']),
            'Wind_Speed': float(request.form['wind_speed']),
            'Humidity': float(request.form['humidity']),
            'Pressure': float(request.form['pressure']),
        }

        # Predict water level
        predicted_water_level = predict_water_level(input_data, loaded_scaler, loaded_model)
        return render_template('result.html', prediction=predicted_water_level)

if __name__ == '__main__':
    app.run(debug=True)
