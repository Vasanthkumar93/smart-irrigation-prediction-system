"""
Smart Irrigation Prediction System - Flask Web Application
Backend Server with Machine Learning Inference, SQLite Storage, and REST API.
"""

import os
import sqlite3
import json
from datetime import datetime
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

app = Flask(__name__)
app.secret_key = 'smart-irrigation-college-secret-key-2026'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DB_PATH = os.path.join(BASE_DIR, 'smart_irrigation.db')

BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
METRICS_PATH = os.path.join(MODELS_DIR, 'model_metrics.json')

# Global references for loaded artifacts
model = None
scaler = None
metrics = {}

def init_db():
    """Initialize SQLite database for recording prediction history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            soil_moisture REAL NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            rainfall REAL NOT NULL,
            prediction INTEGER NOT NULL,
            prediction_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            guidance TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def load_artifacts():
    """Load model, scaler, and metrics metadata. Auto-train if artifacts are missing."""
    global model, scaler, metrics
    if not (os.path.exists(BEST_MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(METRICS_PATH)):
        print("[!] Model artifacts not found. Initiating automatic training...")
        try:
            from train_model import train_and_evaluate
            train_and_evaluate()
        except Exception as e:
            print(f"[ERROR] Auto-training failed: {e}")

    if os.path.exists(BEST_MODEL_PATH):
        model = joblib.load(BEST_MODEL_PATH)
        print(f"[✓] Loaded best model from: {BEST_MODEL_PATH}")
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"[✓] Loaded scaler from: {SCALER_PATH}")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        print(f"[✓] Loaded metrics metadata: {metrics.get('best_model')}")

def get_irrigation_guidance(prediction, sm, temp, hum, rain):
    """
    Generate agronomic advisory and irrigation guidance based on prediction
    and environmental parameters.
    """
    if prediction == 1:
        # Irrigation required
        severity = "High" if sm < 18 else "Moderate"
        water_amount = "25 - 35 mm" if sm < 18 else "15 - 20 mm"
        
        timing = "Early Morning (5:00 AM - 7:30 AM) or Late Evening (6:00 PM - 8:00 PM)"
        if temp > 35:
            timing += " to minimize high daytime evaporation losses."
        else:
            timing += " to optimize soil capillary absorption."

        method = "Drip irrigation recommended to conserve water and prevent foliage fungal diseases."
        if rain > 0:
            weather_note = f"Light rainfall ({rain} mm) observed, but soil moisture ({sm}%) remains below threshold. Proceed with targeted root-zone watering."
        else:
            weather_note = "Zero precipitation recorded. Ensure consistent soil hydration to protect crops from moisture stress."

        guidance = {
            'action': 'Irrigate Now',
            'status_class': 'danger',
            'severity': severity,
            'recommended_water': water_amount,
            'best_timing': timing,
            'recommended_method': method,
            'weather_note': weather_note,
            'badge': 'Irrigation Required'
        }
    else:
        # No irrigation required
        if rain >= 10:
            reason = f"Substantial rainfall ({rain} mm) observed. Soil has received sufficient natural moisture."
            action_text = "Monitor drainage channels to prevent waterlogging and root asphyxiation."
        elif sm >= 50:
            reason = f"Soil moisture is ample ({sm}%). Root zone is well within field capacity."
            action_text = "No watering needed today. Re-evaluate moisture levels in 48 hours."
        else:
            reason = f"Combined environmental conditions (Temp: {temp}°C, Humidity: {hum}%) do not induce acute moisture stress."
            action_text = "Hold off on irrigation. Conserve ground and tank water reserves."

        guidance = {
            'action': 'Do Not Irrigate',
            'status_class': 'success',
            'severity': 'Safe',
            'recommended_water': '0 mm (Conserve Water)',
            'best_timing': 'N/A',
            'recommended_method': 'Maintain regular monitoring; verify soil tensiometer or probes.',
            'weather_note': f"{reason} {action_text}",
            'badge': 'Optimal Moisture'
        }

    return guidance

def validate_inputs(sm_str, temp_str, hum_str, rain_str):
    """Validate numerical and physical boundary constraints."""
    errors = []
    values = {}

    try:
        sm = float(sm_str)
        if not (0.0 <= sm <= 100.0):
            errors.append("Soil Moisture must be between 0% and 100%.")
        values['soil_moisture'] = sm
    except (ValueError, TypeError):
        errors.append("Soil Moisture must be a valid numeric value.")

    try:
        temp = float(temp_str)
        if not (-10.0 <= temp <= 65.0):
            errors.append("Temperature must be between -10°C and 65°C.")
        values['temperature'] = temp
    except (ValueError, TypeError):
        errors.append("Temperature must be a valid numeric value.")

    try:
        hum = float(hum_str)
        if not (0.0 <= hum <= 100.0):
            errors.append("Humidity must be between 0% and 100%.")
        values['humidity'] = hum
    except (ValueError, TypeError):
        errors.append("Humidity must be a valid numeric value.")

    try:
        rain = float(rain_str)
        if not (0.0 <= rain <= 500.0):
            errors.append("Rainfall must be between 0 mm and 500 mm.")
        values['rainfall'] = rain
    except (ValueError, TypeError):
        errors.append("Rainfall must be a valid numeric value.")

    return errors, values

def perform_prediction(sm, temp, hum, rain):
    """Execute prediction using the loaded model and scaler."""
    global model, scaler, metrics
    if model is None or scaler is None:
        load_artifacts()

    if model is None:
        raise RuntimeError("Model artifact could not be loaded. Please run 'python train_model.py'.")

    features = np.array([[sm, temp, hum, rain]])
    
    # Check if the best model relies on scaled features
    best_name = metrics.get('best_model', '')
    uses_scaler = metrics.get('models_evaluated', {}).get(best_name, {}).get('uses_scaler', False)

    if uses_scaler:
        features_to_model = scaler.transform(features)
    else:
        features_to_model = features

    pred = int(model.predict(features_to_model)[0])

    # Probability / Confidence
    if hasattr(model, 'predict_proba'):
        prob_arr = model.predict_proba(features_to_model)[0]
        confidence = float(prob_arr[pred]) * 100.0
    else:
        confidence = 95.0

    guidance = get_irrigation_guidance(pred, sm, temp, hum, rain)

    # Save to SQLite database
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        label_str = "Yes" if pred == 1 else "No"
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, soil_moisture, temperature, humidity, rainfall, prediction, prediction_label, confidence, guidance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (now_str, sm, temp, hum, rain, pred, label_str, round(confidence, 1), guidance['weather_note']))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[!] Warning: Could not write prediction to database: {e}")

    return {
        'prediction': pred,
        'prediction_label': "Irrigation Required (Yes)" if pred == 1 else "Irrigation Not Needed (No)",
        'confidence': round(confidence, 1),
        'inputs': {
            'soil_moisture': sm,
            'temperature': temp,
            'humidity': hum,
            'rainfall': rain
        },
        'guidance': guidance,
        'model_used': metrics.get('best_model', 'Random Forest')
    }

# ----------------- FLASK ROUTES ----------------- #

@app.route('/')
def home():
    """Dashboard landing page."""
    total_predictions = 0
    recent_predictions = []
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]

        cursor.execute("""
            SELECT timestamp, soil_moisture, temperature, humidity, rainfall, prediction_label, confidence 
            FROM predictions ORDER BY id DESC LIMIT 5
        """)
        recent_predictions = cursor.fetchall()
        conn.close()
    except Exception:
        pass

    best_name = metrics.get('best_model', 'Random Forest')
    best_acc = 0.0
    if metrics and 'models_evaluated' in metrics:
        best_acc = metrics['models_evaluated'].get(best_name, {}).get('accuracy', 0.96) * 100

    return render_template('index.html',
                           total_predictions=total_predictions,
                           recent_predictions=recent_predictions,
                           best_model=best_name,
                           best_accuracy=round(best_acc, 2))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page with interactive form and results."""
    result = None
    input_data = {
        'soil_moisture': 18.5,
        'temperature': 34.0,
        'humidity': 32.0,
        'rainfall': 0.0
    }

    if request.method == 'POST':
        sm_val = request.form.get('soil_moisture')
        temp_val = request.form.get('temperature')
        hum_val = request.form.get('humidity')
        rain_val = request.form.get('rainfall')

        errors, values = validate_inputs(sm_val, temp_val, hum_val, rain_val)

        if errors:
            for err in errors:
                flash(err, 'danger')
            # retain user values
            input_data = {
                'soil_moisture': sm_val or '',
                'temperature': temp_val or '',
                'humidity': hum_val or '',
                'rainfall': rain_val or ''
            }
        else:
            try:
                result = perform_prediction(
                    values['soil_moisture'],
                    values['temperature'],
                    values['humidity'],
                    values['rainfall']
                )
                input_data = values
                flash("Prediction calculated successfully based on environmental conditions!", "success")
            except Exception as e:
                flash(f"Error executing prediction: {str(e)}", "danger")

    return render_template('predict.html', result=result, inputs=input_data)

@app.route('/models')
def models_view():
    """Model comparison and performance evaluation page."""
    evaluated = metrics.get('models_evaluated', {})
    best_name = metrics.get('best_model', 'Random Forest')
    features = metrics.get('features', ['soil_moisture', 'temperature', 'humidity', 'rainfall'])
    feat_importances = metrics.get('feature_importance', {})

    return render_template('models.html',
                           models=evaluated,
                           best_model=best_name,
                           features=features,
                           feature_importances=feat_importances)

@app.route('/history')
def history():
    """Historical records stored in SQLite database."""
    records = []
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, timestamp, soil_moisture, temperature, humidity, rainfall, 
                   prediction, prediction_label, confidence, guidance 
            FROM predictions ORDER BY id DESC
        ''')
        records = cursor.fetchall()
        conn.close()
    except Exception as e:
        flash(f"Could not load history: {e}", "danger")

    return render_template('history.html', records=records)

@app.route('/history/clear', methods=['POST'])
def clear_history():
    """Clear past prediction logs."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        flash("Prediction history cleared successfully.", "info")
    except Exception as e:
        flash(f"Failed to clear history: {e}", "danger")
    return redirect(url_for('history'))

@app.route('/about')
def about():
    """Documentation and agronomic theoretical background."""
    return render_template('about.html')

# ----------------- REST API ENDPOINT ----------------- #

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    REST API endpoint for IoT / Mobile / Automated integration.
    Accepts JSON: {"soil_moisture": float, "temperature": float, "humidity": float, "rainfall": float}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'status': 'error', 'message': 'Invalid JSON or missing body.'}), 400

    errors, values = validate_inputs(
        data.get('soil_moisture'),
        data.get('temperature'),
        data.get('humidity'),
        data.get('rainfall')
    )

    if errors:
        return jsonify({'status': 'error', 'errors': errors}), 422

    try:
        res = perform_prediction(
            values['soil_moisture'],
            values['temperature'],
            values['humidity'],
            values['rainfall']
        )
        return jsonify({'status': 'success', 'data': res}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ----------------- APPLICATION ENTRYPOINT ----------------- #

# Initialize DB on load
init_db()
load_artifacts()

if __name__ == '__main__':
    # Local development server for VS Code
    app.run(host='127.0.0.1', port=5000, debug=True)
