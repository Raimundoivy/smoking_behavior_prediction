import pytest
import pandas as pd
import joblib
import os
from prediction_service import PredictionService

@pytest.fixture
def reset_prediction_service_singleton():
    PredictionService._instance = None
    yield
    PredictionService._instance = None # Reset after test as well

# Test for successful instance creation and prediction
def test_prediction_service_predict_success(mocker, reset_prediction_service_singleton):
    # Create a mock for the pipeline model
    mock_pipeline_model = mocker.Mock()
    mock_pipeline_model.predict.return_value = [0] # Example prediction
    mock_pipeline_model.predict_proba.return_value = [[0.2, 0.8]] # Example probabilities
    
    # Patch joblib.load to return our mock pipeline model
    mocker.patch('joblib.load', return_value=mock_pipeline_model)

    service = PredictionService(model_path="dummy_path.joblib")
    
    # Now, replace the loaded model with our mock
    service.model = mock_pipeline_model

    assert service is not None
    assert service.model is not None

    input_data = {
        'age': 30,
        'gender': 'Male',
        'marital_status': 'Single',
        'highest_qualification': 'Degree',
        'nationality': 'British',
        'ethnicity': 'White',
        'gross_income': 'Above 36,400',
        'region': 'South East'
    }
    prediction, probability = service.predict(input_data)
    assert prediction == 0
    assert probability == 0.8
    mock_pipeline_model.predict.assert_called_once()
    mock_pipeline_model.predict_proba.assert_called_once()

# Test for FileNotFoundError during model loading
def test_prediction_service_model_not_found(mocker, reset_prediction_service_singleton):
    # Patch joblib.load to raise FileNotFoundError when called
    mocker.patch('joblib.load', side_effect=FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        PredictionService(model_path="non_existent_path.joblib")

# Test for other Exceptions during model loading
def test_prediction_service_model_corrupted(mocker, reset_prediction_service_singleton):
    # Patch joblib.load to raise a generic Exception when called
    mocker.patch('joblib.load', side_effect=Exception("Corrupted file"))
    with pytest.raises(RuntimeError, match="An error occurred while loading the model: Corrupted file"):
        PredictionService(model_path="corrupted_path.joblib")
