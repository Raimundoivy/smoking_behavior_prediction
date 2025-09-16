import logging
from flask import Flask, render_template, request, flash, redirect, url_for
from prediction_service import prediction_service
import config

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config.from_object(config)

@app.route('/')
def home():
    return render_template("index.html",
                           genders=config.VALID_GENDERS,
                           marital_statuses=config.VALID_MARITAL_STATUSES,
                           qualifications=config.VALID_QUALIFICATIONS,
                           nationalities=config.VALID_NATIONALITIES,
                           ethnicities=config.VALID_ETHNICITIES,
                           incomes=config.VALID_INCOMES,
                           regions=config.VALID_REGIONS)

@app.route('/predict', methods=['POST'])
def predict():
    required_fields = ['age', 'gender', 'marital_status', 'highest_qualification', 'nationality', 'ethnicity', 'gross_income', 'region']
    if not all(field in request.form and request.form[field] for field in required_fields):
        flash("All fields are required.", "danger")
        return redirect(url_for('home'))
    
    try:
        # Get the input from the form
        age = int(request.form['age'])
        if not (0 < age < 120):
            flash("Invalid input: Please enter an age between 1 and 120.", "danger")
            return redirect(url_for('home'))
        
        gender = request.form['gender']
        if gender not in config.VALID_GENDERS:
            flash(f"Invalid gender. Please choose from {', '.join(config.VALID_GENDERS)}.", "danger")
            return redirect(url_for('home'))

        marital_status = request.form['marital_status']
        if marital_status not in config.VALID_MARITAL_STATUSES:
            flash(f"Invalid marital status. Please choose from {', '.join(config.VALID_MARITAL_STATUSES)}.", "danger")
            return redirect(url_for('home'))

        highest_qualification = request.form['highest_qualification']
        if highest_qualification not in config.VALID_QUALIFICATIONS:
            flash(f"Invalid highest qualification. Please choose from {', '.join(config.VALID_QUALIFICATIONS)}.", "danger")
            return redirect(url_for('home'))

        nationality = request.form['nationality']
        if nationality not in config.VALID_NATIONALITIES:
            flash(f"Invalid nationality. Please choose from {', '.join(config.VALID_NATIONALITIES)}.", "danger")
            return redirect(url_for('home'))

        ethnicity = request.form['ethnicity']
        if ethnicity not in config.VALID_ETHNICITIES:
            flash(f"Invalid ethnicity. Please choose from {', '.join(config.VALID_ETHNICITIES)}.", "danger")
            return redirect(url_for('home'))
        if ethnicity not in config.VALID_ETHNICITIES:
            flash(f"Invalid ethnicity. Please choose from {', '.join(config.VALID_ETHNICITIES)}.", "danger")
            return redirect(url_for('home'))

        gross_income = request.form['gross_income']
        if gross_income not in config.VALID_INCOMES:
            flash(f"Invalid gross income. Please choose from {', '.join(config.VALID_INCOMES)}.", "danger")
            return redirect(url_for('home'))

        region = request.form['region']
        if region not in config.VALID_REGIONS:
            flash(f"Invalid region. Please choose from {', '.join(config.VALID_REGIONS)}.", "danger")
            return redirect(url_for('home'))

        input_data = {
            'age': age,
            'gender': gender,
            'marital_status': marital_status,
            'highest_qualification': highest_qualification,
            'nationality': nationality,
            'ethnicity': ethnicity,
            'gross_income': gross_income,
            'region': region
        }

        # Make prediction using the service
        prediction, smoking_probability = prediction_service.predict(input_data)

        return render_template("result.html", prediction=prediction, smoking_probability=smoking_probability)
    
    except ValueError:
        logging.warning(f"Invalid input for age: {request.form.get('age')}")
        flash("Invalid input: Please enter a valid number for age.", "danger")
        return redirect(url_for('home'))
    
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}", exc_info=True)
        flash("An unexpected error occurred. Please try again later.", "danger")
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=config.FLASK_DEBUG)