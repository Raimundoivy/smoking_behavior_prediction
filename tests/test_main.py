import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from main import load_data, define_pipeline, train_model
import os
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors

# Fixture for a dummy CSV file
@pytest.fixture
def dummy_csv(tmp_path):
    content = """,age,gender,marital_status,highest_qualification,nationality,ethnicity,gross_income,region,smoke,amt_weekends,amt_weekdays,type
1,30,Male,Single,Degree,British,White,"Above 36,400",South East,Yes,10,5,A
2,25,Female,Married,GCSE/CSE,English,Mixed,"2,600 to 5,200",The North,No,8,4,B
3,40,Male,Divorced,A Levels,Irish,Black,"28,600 to 36,400",Midlands & East Anglia,Yes,12,6,C
4,35,Female,Separated,ONC/BTEC,Scottish,Chinese,"10,400 to 15,600",South West,No,9,3,D
5,50,Male,Widowed,Other/Sub Degree,Welsh,Asian,"Above 36,400",Scotland,Yes,15,7,E
6,28,Female,Single,GCSE/O Level,British,White,"Under 2,600",South East,No,7,2,F
7,45,Male,Married,Degree,English,Mixed,"5,200 to 10,400",The North,No,11,5,G
"""
    file_path = tmp_path / "test_smoking.csv"
    file_path.write_text(content)
    return str(file_path)

def test_load_data(dummy_csv):
    df = load_data(dummy_csv)
    assert isinstance(df, pd.DataFrame)
    assert 'amt_weekends' not in df.columns
    assert 'amt_weekdays' not in df.columns
    assert 'type' not in df.columns
    assert 'age' in df.columns
    assert 'smoke' in df.columns
    assert 'gender' in df.columns
    assert 'marital_status' in df.columns
    assert 'highest_qualification' in df.columns
    assert 'nationality' in df.columns
    assert 'ethnicity' in df.columns
    assert 'gross_income' in df.columns
    assert 'region' in df.columns
    assert len(df) == 7


def test_define_pipeline():
    pipeline = define_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert 'preprocessor' in pipeline.named_steps
    assert 'smote' in pipeline.named_steps
    assert 'classifier' in pipeline.named_steps

# Mock the joblib.dump to prevent actual file saving during test
def test_train_model(dummy_csv, mocker):
    mocker.patch('joblib.dump')
    
    # Patch SMOTE used by main.py
    mock_smote_instance = SMOTE(random_state=42, k_neighbors=1)
    mocker.patch('main.SMOTE', return_value=mock_smote_instance)

    df = load_data(dummy_csv)
    pipeline = define_pipeline()
    trained_pipeline = train_model(pipeline, df)
    assert isinstance(trained_pipeline, Pipeline)
    assert hasattr(trained_pipeline.named_steps['classifier'], 'coef_')
