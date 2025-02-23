from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)

# Define model parameters
FEATURES = ['CHLA_SAT', 'BBP_DCM', 'D26Depth']
TARGET = 'DCM_depth'
DATA_FILE = 'extracted_data.xlsx'

# Model file paths
MODEL_FILES = {
    "random_forest": "rf_model.pkl",
    "neural_network": "nn_model.h5",
    "svm": "svm_model.pkl",
    "decision_tree": "dt_model.pkl"
}

# Scaler file paths (only for models that require scaling)
SCALER_FILES = {
    "neural_network": {"mean": "scaler_mean.npy", "scale": "scaler_scale.npy"},
    "svm": "scaler.pkl"
}

# Load dataset
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Dataset '{DATA_FILE}' not found!")

data_table = pd.read_excel(DATA_FILE, engine='openpyxl')

# Extract features and target variable
X = data_table[FEATURES].values
Y = data_table[TARGET].values

# Remove rows with NaN values
valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(Y)
X = X[valid_idx]
Y = Y[valid_idx]

# Load models
models = {}
for model_name, model_file in MODEL_FILES.items():
    if os.path.exists(model_file):
        if model_name == "neural_network":
            models[model_name] = keras.models.load_model(model_file)
        else:
            models[model_name] = joblib.load(model_file)
    else:
        print(f"Warning: {model_name} model file '{model_file}' not found!")

# Load scalers for models that require scaling
scalers = {}
if os.path.exists(SCALER_FILES["svm"]):
    scalers["svm"] = joblib.load(SCALER_FILES["svm"])
if os.path.exists(SCALER_FILES["neural_network"]["mean"]) and os.path.exists(SCALER_FILES["neural_network"]["scale"]):
    scalers["neural_network"] = {
        "mean": np.load(SCALER_FILES["neural_network"]["mean"]),
        "scale": np.load(SCALER_FILES["neural_network"]["scale"])
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_type = data.get("model_type")  # Expecting model_type in request
        
        if model_type not in models:
            return jsonify({"error": "Invalid model_type. Choose from random_forest, neural_network, svm, decision_tree."}), 400

        # Extract input features
        input_features = np.array([[data.get(feature, 0.0) for feature in FEATURES]])

        if input_features.shape[1] != len(FEATURES):
            return jsonify({'error': f"Expected {len(FEATURES)} features, but got {input_features.shape[1]}"}), 400

        # Apply feature scaling if required
        if model_type in scalers:
            if model_type == "neural_network":
                mean_ = scalers["neural_network"]["mean"]
                scale_ = scalers["neural_network"]["scale"]
                input_features = (input_features - mean_) / scale_
            elif model_type == "svm":
                input_features = scalers["svm"].transform(input_features)
        
        # Make prediction
        if model_type == "neural_network":
            prediction = models[model_type].predict(input_features)[0][0]
        else:
            prediction = models[model_type].predict(input_features)[0]

        return jsonify({'predicted_dcm_depth': float(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'Model API is running'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
