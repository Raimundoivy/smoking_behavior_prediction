import logging
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from prediction_service import get_prediction_service
import config

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

# --- Load Model and Interpretability Components on Startup ---
try:
    prediction_service = get_prediction_service()
    logging.info("Prediction service loaded successfully.")
    
    preprocessor = prediction_service.model.named_steps['preprocessor']
    classifier = prediction_service.model.named_steps['classifier']
    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    age_scaler_mean = preprocessor.named_transformers_['num'].mean_[0]

except Exception as e:
    logging.error(f"Failed to load prediction service or components: {e}", exc_info=True)
    prediction_service = None

def get_narrative(feature_name, user_age):
    """Generates a human-readable explanation for a feature name."""
    parts = feature_name.replace('__', ' ').split()
    if parts[0] == 'num age':
        comparison = "higher" if user_age > age_scaler_mean else "lower"
        return f"Age being {comparison} than average"
    category = parts[1].replace('_', ' ').title()
    value = " ".join(parts[2:])
    return f"{category} being '{value}'"

def calculate_confidence(probability):
    """Calculates a qualitative confidence score based on the prediction probability."""
    distance_from_center = abs(probability - 0.5)
    if distance_from_center > 0.25: # e.g., <25% or >75%
        return "High"
    elif distance_from_center > 0.1: # e.g., <40% or >60%
        return "Medium"
    else:
        return "Low"

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests, returning results with confidence and local feature importance."""
    if prediction_service is None: return jsonify({"error": "Model is not available"}), 503
    if not request.is_json: return jsonify({"error": "Request must be JSON"}), 400

    input_data = request.get_json()
    
    required_keys = ['age', 'gender', 'marital_status', 'highest_qualification', 'nationality', 'ethnicity', 'gross_income', 'region']
    if not all(key in input_data for key in required_keys): return jsonify({"error": "Missing required fields"}), 400

    try:
        input_df = pd.DataFrame([input_data])
        prediction, smoking_probability = prediction_service.predict(input_data)
        
        transformed_features = preprocessor.transform(input_df)[0]
        contributions = transformed_features * coefficients
        confidence = calculate_confidence(smoking_probability)
        
        top_indices = np.argsort(np.abs(contributions))[-5:][::-1]
        
        feature_importance = []
        for i in top_indices:
            if abs(contributions[i]) < 1e-6: continue
            feature_importance.append({
                'narrative': get_narrative(feature_names[i], input_data['age']),
                'contribution': contributions[i],
                'direction': 'positive' if contributions[i] > 0 else 'negative'
            })

        response = {'prediction': int(prediction), 'smoking_probability': float(smoking_probability), 'confidence': confidence, 'feature_importance': feature_importance}
        return jsonify(response)
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred"}), 500

@app.route('/global-importance', methods=['GET'])
def global_importance():
    """Returns the top overall features the model considers important."""
    if prediction_service is None: return jsonify({"error": "Model is not available"}), 503
    
    global_importances = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
    top_5 = [{'feature': get_narrative(f[0], age_scaler_mean)} for f in global_importances[:5]]
    return jsonify(top_5)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=config.FLASK_DEBUG)