import logging
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from prediction_service import get_prediction_service
import config

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

try:
    prediction_service = get_prediction_service()
    logging.info("Prediction service loaded successfully.")
    
    # Extract feature names and coefficients from the pipeline for interpretability
    preprocessor = prediction_service.model.named_steps['preprocessor']
    classifier = prediction_service.model.named_steps['classifier']
    feature_names = preprocessor.get_feature_names_out()
    coefficients = classifier.coef_[0]
    
except Exception as e:
    logging.error(f"Failed to load prediction service or components: {e}", exc_info=True)
    prediction_service = None

@app.route('/predict', methods=['POST'])
def predict():
    if prediction_service is None:
        return jsonify({"error": "Model is not available"}), 503

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    input_data = request.get_json()
    
    required_keys = ['age', 'gender', 'marital_status', 'highest_qualification', 'nationality', 'ethnicity', 'gross_income', 'region']
    if not all(key in input_data for key in required_keys):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        input_df = pd.DataFrame([input_data])
        prediction, smoking_probability = prediction_service.predict(input_data)
        
        # --- Feature Importance Logic ---
        # 1. Transform the user's input using the same preprocessor
        transformed_features = preprocessor.transform(input_df)[0]
        
        # 2. Calculate the contribution of each feature
        contributions = transformed_features * coefficients
        
        # 3. Get the top 3 most influential features
        top_indices = np.argsort(np.abs(contributions))[-3:][::-1]
        
        feature_importance = []
        for i in top_indices:
            feature_importance.append({
                'feature': feature_names[i].replace('cat__', '').replace('ord__', '').replace('num__', '').replace('_', ' '),
                'contribution': contributions[i]
            })

        response = {
            'prediction': int(prediction),
            'smoking_probability': float(smoking_probability),
            'feature_importance': feature_importance
        }
        return jsonify(response)

    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=config.FLASK_DEBUG)