# Smoking Behavior Prediction

This project is a machine learning application that predicts whether a person is a smoker based on their demographic and socio-economic information. The project includes a machine learning pipeline to process the data and train a model, and a Flask web application to serve the model and provide predictions.

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd smoking-behavior-prediction
    ```

2.  **Create and activate a Python virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Execution

1.  **Run the training pipeline:**

    To train the model and generate the `full_pipeline.joblib` artifact, run the following command:

    ```bash
    python main.py
    ```

2.  **Start the Flask web application:**

    To start the Flask web application, run the following command:

    ```bash
    python app.py
    ```

    The application will be available at `http://127.0.0.1:5000`.

## Project Structure

-   `main.py`: This script contains the code for building and training the machine learning pipeline. It loads the data, preprocesses it, trains a logistic regression model, and saves the entire pipeline as a `.joblib` file.
-   `app.py`: This is the main file for the Flask web application. It loads the saved pipeline and provides a web interface for users to input their information and get a prediction.
-   `full_pipeline.joblib`: This is the saved machine learning pipeline artifact. It includes all the preprocessing steps (encoding, scaling), the SMOTE resampling step, and the trained logistic regression model.
-   `templates/`: This directory contains the HTML templates for the web application.

## Dataset

The dataset used for training is `smoking.csv`. It contains various demographic and socio-economic features of individuals, such as age, gender, marital status, highest qualification, nationality, ethnicity, gross income, and region. The target variable is `smoke`, which indicates whether a person is a smoker or not.

## Machine Learning Pipeline

The machine learning pipeline is built using `scikit-learn` and `imblearn` and consists of the following steps:

1.  **Preprocessing:**
    -   **Categorical Features:** `OneHotEncoder` is used to convert categorical features into a numerical format.
    -   **Ordinal Features:** `OrdinalEncoder` is used to convert ordinal features into a numerical format.
    -   **Numerical Features:** `StandardScaler` is used to scale the numerical features.
2.  **Resampling:**
    -   `SMOTE` (Synthetic Minority Over-sampling Technique) is used to handle the class imbalance in the dataset by generating synthetic samples for the minority class.
3.  **Model:**
    -   A `LogisticRegression` model is used as the final classifier to predict whether a person is a smoker or not.