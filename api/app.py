# api/app.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import traceback
from ml.utils import transform_amount_with_scaler

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/xgb_model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '../models/scaler.joblib')

MODEL_PATH = os.path.abspath(MODEL_PATH)
SCALER_PATH = os.path.abspath(SCALER_PATH)

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

REQUIRED = set(['Time','Amount'] + [f'V{i}' for i in range(1,29)])

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'No JSON received'}), 400
        df = pd.DataFrame(data if isinstance(data, list) else [data])

        missing = REQUIRED - set(df.columns)
        if missing:
            return jsonify({'error': f'Missing columns: {sorted(list(missing))}'}), 400

        df_t = transform_amount_with_scaler(df, scaler)
        probs = model.predict_proba(df_t)[:,1]
        df['fraud_probability'] = probs
        result = df.to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        f = request.files['file']
        df = pd.read_csv(f)
        missing = REQUIRED - set(df.columns)
        if missing:
            return jsonify({'error': f'Missing columns: {sorted(list(missing))}'}), 400
        df_t = transform_amount_with_scaler(df, scaler)
        probs = model.predict_proba(df_t)[:,1]
        df['fraud_probability'] = probs
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
