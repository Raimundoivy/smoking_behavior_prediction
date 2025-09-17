import joblib
import pandas as pd
import config

class PredictionService:
    """A service to load a model and make predictions. This is a plain class."""
    
    def __init__(self, model_path: str):
        """
        Initializes the service by loading the model from the given path.
        """
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Loads the model from the specified path."""
        try:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please ensure the model is trained and saved."
            ) from e
        except Exception as e:
            raise RuntimeError(f"An error occurred while loading the model: {e}") from e

    def predict(self, input_data: dict):
        """
        Makes a prediction based on the input data.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please check the model path and file integrity.")
        
        input_df = pd.DataFrame([input_data])
        prediction = self.model.predict(input_df)[0]
        probabilities = self.model.predict_proba(input_df)[0]
        smoking_probability = probabilities[1] 
        return prediction, smoking_probability

# --- Factory and Global Instance Management ---

_prediction_service_instance = None

def get_prediction_service():
    """
    Factory function to get a singleton instance of the PredictionService.
    This function controls WHEN the instance is created.
    """
    global _prediction_service_instance
    if _prediction_service_instance is None:
        print("Creating new PredictionService instance via factory.")
        _prediction_service_instance = PredictionService(model_path=config.MODEL_PATH)
    return _prediction_service_instance