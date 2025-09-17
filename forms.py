from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, NumberRange
import config

class PredictionForm(FlaskForm):
    """Form for collecting user input for smoking prediction."""

    age = IntegerField(
        'Age',
        validators=[
            DataRequired(message="Age is a required field."),
            NumberRange(min=1, max=120, message="Please enter a valid age between 1 and 120.")
        ]
    )
    
    gender = SelectField(
        'Gender',
        choices=config.VALID_GENDERS,
        validators=[DataRequired()]
    )
    
    marital_status = SelectField(
        'Marital Status',
        choices=config.VALID_MARITAL_STATUSES,
        validators=[DataRequired()]
    )
    
    highest_qualification = SelectField(
        'Highest Qualification',
        choices=config.VALID_QUALIFICATIONS,
        validators=[DataRequired()]
    )
    
    nationality = SelectField(
        'Nationality',
        choices=config.VALID_NATIONALITIES,
        validators=[DataRequired()]
    )
    
    ethnicity = SelectField(
        'Ethnicity',
        choices=config.VALID_ETHNICITIES,
        validators=[DataRequired()]
    )
    
    gross_income = SelectField(
        'Gross Income',
        choices=config.VALID_INCOMES,
        validators=[DataRequired()]
    )
    
    region = SelectField(
        'Region',
        choices=config.VALID_REGIONS,
        validators=[DataRequired()]
    )

    submit = SubmitField('Predict')