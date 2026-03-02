import os
import pickle
import pandas as pd


def load_model_and_predict(df, model_path="models/model.pkl"):
    # Resolve absolute paths relative to this script if a relative path is provided
    if not os.path.isabs(model_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, model_path)

    # Prepare features
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    # Load trained model pipeline
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # The pipeline handles all preprocessing
    churn_probs = model.predict_proba(X)[:, 1]
    return churn_probs
