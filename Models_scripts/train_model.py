import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from data_processing import load_and_clean_data


def train_and_save_model():
    # Resolve absolute paths relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "CustomerChurn.csv")
    model_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(model_dir, "model.pkl")

    # Load data
    df = load_and_clean_data(data_path)

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    # Split data to evaluate properly
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['number']).columns

    # Build the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    # Create the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print(f"Model Evaluation on Test Set:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

    os.makedirs(model_dir, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"Model trained and saved to {model_path}")


if __name__ == "__main__":
    train_and_save_model()