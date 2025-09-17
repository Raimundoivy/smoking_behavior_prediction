import logging
import os
import requests
from flask import Flask, render_template, flash
import config
from forms import PredictionForm

# Ensure the secret key is set before running the app
if not config.SECRET_KEY:
    raise ValueError("SECRET_KEY not set in environment. Please set it in your .env file.")

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config.from_object(config)

# The URL for our new API service.
# We use the service name 'api' from docker-compose, which Docker resolves to the correct container IP.
API_URL = "http://api:8000/predict"

@app.route('/', methods=['GET', 'POST'])
def home():
    form = PredictionForm()
    if form.validate_on_submit():
        # Collect data from the form object.
        input_data = {
            'age': form.age.data,
            'gender': form.gender.data,
            'marital_status': form.marital_status.data,
            'highest_qualification': form.highest_qualification.data,
            'nationality': form.nationality.data,
            'ethnicity': form.ethnicity.data,
            'gross_income': form.gross_income.data,
            'region': form.region.data
        }

        try:
            # Make a POST request to our API service
            response = requests.post(API_URL, json=input_data, timeout=5)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Parse the JSON response from the API
            result = response.json()
            prediction = result['prediction']
            smoking_probability = result['smoking_probability']

            return render_template("result.html", prediction=prediction, smoking_probability=smoking_probability)

        except requests.exceptions.RequestException as e:
            logging.error(f"Could not connect to API service: {e}", exc_info=True)
            flash("The prediction service is currently unavailable. Please try again later.", "danger")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}", exc_info=True)
            flash("An unexpected error occurred. Please try again later.", "danger")
            
    # If GET request or validation fails, render the index page with the form
    return render_template("index.html", form=form)

if __name__ == '__main__':
    app.run(debug=config.FLASK_DEBUG)