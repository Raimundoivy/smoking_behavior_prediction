import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import config

def load_data(file_path):
    """Loads data from a CSV file and drops unnecessary columns."""
    print("Loading data...")
    df = pd.read_csv(file_path, index_col=0)
    df = df.drop(['amt_weekends', 'amt_weekdays', 'type'], axis=1)
    return df

def define_pipeline():
    """Defines and returns the machine learning pipeline."""
    print("Defining pipeline...")
    categorical_features = ['gender', 'marital_status', 'nationality', 'ethnicity', 'region']
    ordinal_features = ['highest_qualification', 'gross_income']
    numerical_features = ['age']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('ord', OrdinalEncoder(categories=[config.VALID_QUALIFICATIONS, config.VALID_INCOMES]), ordinal_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    model = LogisticRegression(random_state=42, max_iter=1000)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('smote', SMOTE(random_state=42)),
                                ('classifier', model)])
    return pipeline

def train_model(pipeline, df):
    """Trains the model and returns the trained pipeline."""
    print("Training the model...")
    X = df.drop('smoke', axis=1)
    y = df['smoke'].apply(lambda x: 1 if x == 'Yes' else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)

    print("Evaluating the model...")
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}\n")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return pipeline

def save_pipeline(pipeline, file_path):
    """Saves the pipeline to a joblib file."""
    print("Saving the full pipeline...")
    joblib.dump(pipeline, file_path)
    print("Pipeline saved successfully!")

if __name__ == "__main__":
    df = load_data(config.DATA_PATH)
    pipeline = define_pipeline()
    trained_pipeline = train_model(pipeline, df)
    save_pipeline(trained_pipeline, config.MODEL_PATH)
