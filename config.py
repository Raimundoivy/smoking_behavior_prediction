import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

MODEL_PATH = "full_pipeline.joblib"
DATA_PATH = "smoking.csv"
SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY:
    raise ValueError("SECRET_KEY not set. Please set it as an environment variable.")
FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'

# Define valid categories for validation
VALID_GENDERS = ['Male', 'Female']
VALID_MARITAL_STATUSES = ['Single', 'Married', 'Divorced', 'Separated', 'Widowed']
VALID_QUALIFICATIONS = ['No Qualification', 'GCSE/CSE', 'GCSE/O Level', 'A Levels', 'ONC/BTEC', 'Other/Sub Degree', 'Higher/Sub Degree', 'Degree', 'Unknown']
VALID_NATIONALITIES = ['British', 'English', 'Irish', 'Scottish', 'Welsh', 'Other', 'Refused', 'Unknown']
VALID_ETHNICITIES = ['White', 'Mixed', 'Black', 'Chinese', 'Asian', 'Refused', 'Unknown']
VALID_INCOMES = ['Under 2,600', '2,600 to 5,200', '5,200 to 10,400', '10,400 to 15,600', '15,600 to 20,800', '20,800 to 28,600', '28,600 to 36,400', 'Above 36,400', 'Refused', 'Unknown']
VALID_REGIONS = ['The North', 'Midlands & East Anglia', 'South East', 'South West', 'Wales', 'Scotland']
