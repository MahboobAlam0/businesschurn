import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_processing import load_and_clean_data


def train_and_save_model():
    # Load data
    df = load_and_clean_data("data/Customer Churn.csv")

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"]

    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    os.makedirs("models", exist_ok=True)

    with open("models/churn_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved to models/churn_model.pkl")


if __name__ == "__main__":
    train_and_save_model()
