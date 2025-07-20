from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ✅ Load all models (use raw strings for Windows paths)
model_sleep = joblib.load(r"C:\Users\anshi\Downloads\MyHealthMap\model_sleep.pkl")
model_heart = joblib.load(r"C:\Users\anshi\Downloads\MyHealthMap\model_heart.pkl")
model_bmi = joblib.load(r"C:\Users\anshi\Downloads\MyHealthMap\model_bmi.pkl")
model_diastolic_bp = joblib.load(r"C:\Users\anshi\Downloads\MyHealthMap\model_diastolic_bp.pkl")
model_systolic_bp = joblib.load(r"C:\Users\anshi\Downloads\MyHealthMap\model_systolic_bp.pkl")
model_stress = joblib.load(r"C:\Users\anshi\Downloads\MyHealthMap\model_stress.pkl")
model_qos = joblib.load(r"C:\Users\anshi\Downloads\MyHealthMap\model_qos.pkl")

@app.route('/predict-all', methods=['POST'])
def predict_all():
    data = request.get_json()

    # ✅ Feature order must match training features
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

    # ✅ Mapping for sleep disorder
    sleep_label_map = {0: "Insomnia", 1: "None", 2: "Sleep Apnea"}

    result = {
        'Sleep Disorder': sleep_label_map.get(sleep_pred, "Unknown"),
        'Heart Rate Category': heart_pred,  # 0: Low, 1: Normal, 2: High
        'BMI Category': bmi_pred,          # 1: Normal, 2: Overweight, 3: Obese
        'Predicted Systolic BP': round(sys_pred, 1),
        'Predicted Diastolic BP': round(dia_pred, 1),
        'Predicted Stress Level': round(stress_pred, 1),
        'Predicted Quality of Sleep': round(qos_pred, 1)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5001)

