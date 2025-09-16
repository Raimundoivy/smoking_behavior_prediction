import joblib
import pandas as pd
import config

class PredictionService:
    _instance = None

    def __new__(cls, model_path=config.MODEL_PATH):
        if cls._instance is None:
            print("Creating new PredictionService instance")
            cls._instance = super(PredictionService, cls).__new__(cls)
            try:
                cls._instance.model = joblib.load(model_path)
                print(f"Model loaded successfully from {model_path}")
            except FileNotFoundError:
                raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the model is trained and saved.")
            except Exception as e:
                raise RuntimeError(f"An error occurred while loading the model: {e}") from e
        return cls._instance

    def predict(self, input_data: dict):
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please check the model path and file integrity.")
        
        input_df = pd.DataFrame([input_data])
        prediction = self.model.predict(input_df)[0]
        probabilities = self.model.predict_proba(input_df)[0]
        smoking_probability = probabilities[1]
        return prediction, smoking_probability

# Create a single, shared instance of the service
prediction_service = PredictionService()
