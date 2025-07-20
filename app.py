from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ✅ Base directory (for safe relative paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Load models
model_sleep = joblib.load(os.path.join(BASE_DIR, "model_sleep.pkl"))
model_heart = joblib.load(os.path.join(BASE_DIR, "model_heart.pkl"))
model_bmi = joblib.load(os.path.join(BASE_DIR, "model_bmi.pkl"))
model_diastolic_bp = joblib.load(os.path.join(BASE_DIR, "model_diastolic_bp.pkl"))
model_systolic_bp = joblib.load(os.path.join(BASE_DIR, "model_systolic_bp.pkl"))
model_stress = joblib.load(os.path.join(BASE_DIR, "model_stress.pkl"))
model_qos = joblib.load(os.path.join(BASE_DIR, "model_qos.pkl"))

@app.route('/predict-all', methods=['POST'])
def predict_all():
    data = request.get_json()

    # ✅ Ensure correct feature order
    features = [
        float(data['Age']),
        1 if data['Gender'].lower() == 'male' else 0,
        float(data['Sleep Duration']),
        float(data['Quality of Sleep']),
        float(data['Physical Activity Level']),
        float(data['Stress Level']),
        float(data['Heart Rate']),
        float(data['Daily Steps'])
    ]

    features = np.array(features).reshape(1, -1)

    # ✅ Make predictions
    sleep_pred = int(model_sleep.predict(features)[0])
    heart_pred = int(model_heart.predict(features)[0])
    bmi_pred = int(model_bmi.predict(features)[0])
    dia_pred = float(model_diastolic_bp.predict(features)[0])
    sys_pred = float(model_systolic_bp.predict(features)[0])
    stress_pred = float(model_stress.predict(features)[0])
    qos_pred = float(model_qos.predict(features)[0])

    # ✅ Mappings
    sleep_label_map = {0: "Insomnia", 1: "None", 2: "Sleep Apnea"}
    heart_label_map = {0: "Low", 1: "Normal", 2: "High"}
    bmi_label_map = {1: "Normal", 2: "Overweight", 3: "Obese"}

    result = {
        'Sleep Disorder': sleep_label_map.get(sleep_pred, "Unknown"),
        'Heart Rate Category': heart_label_map.get(heart_pred, "Unknown"),
        'BMI Category': bmi_label_map.get(bmi_pred, "Unknown"),
        'Predicted Systolic BP': round(sys_pred, 1),
        'Predicted Diastolic BP': round(dia_pred, 1),
        'Predicted Stress Level': round(stress_pred, 1),
        'Predicted Quality of Sleep': round(qos_pred, 1)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
