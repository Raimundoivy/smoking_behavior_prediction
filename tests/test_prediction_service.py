import pytest
from unittest.mock import Mock
from prediction_service import PredictionService # Import the class, not an instance

# Test for successful instance creation and prediction
def test_prediction_service_predict_success(mocker):
    # 1. Mock the file loading function BEFORE the service is instantiated
    mock_pipeline_model = Mock()
    mock_pipeline_model.predict.return_value = [0]
    mock_pipeline_model.predict_proba.return_value = [[0.2, 0.8]]
    mocker.patch('joblib.load', return_value=mock_pipeline_model)

    # 2. Now, instantiate the service. It will use the mocked joblib.load
    service = PredictionService(model_path="dummy_path.joblib")
    
    # 3. Assert and test as before
    assert service.model is not None
    input_data = {'age': 30, 'gender': 'Male', 'marital_status': 'Single', 'highest_qualification': 'Degree', 'nationality': 'British', 'ethnicity': 'White', 'gross_income': 'Above 36,400', 'region': 'South East'}
    prediction, probability = service.predict(input_data)

    assert prediction == 0
    assert probability == 0.8

    mock_pipeline_model.predict_proba.assert_called_once()

# Test for FileNotFoundError during model loading
def test_prediction_service_model_not_found(mocker):
    mocker.patch('joblib.load', side_effect=FileNotFoundError)
    
    # Test the class's __init__ directly
    with pytest.raises(FileNotFoundError):
        PredictionService(model_path="non_existent_path.joblib")

# Test for other Exceptions during model loading
def test_prediction_service_model_corrupted(mocker):
    mocker.patch('joblib.load', side_effect=Exception("Corrupted file"))
    
    with pytest.raises(RuntimeError, match="An error occurred while loading the model: Corrupted file"):
        PredictionService(model_path="corrupted_path.joblib")