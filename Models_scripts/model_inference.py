import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_model_and_predict(df, model_path="models/churn_model.pkl"):
    # Prepare features
    X = df.drop(columns=["Churn", "customerID"])
    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # OK for demo; note below

    # Load trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    churn_probs = model.predict_proba(X_scaled)[:, 1]
    return churn_probs
