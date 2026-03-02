import os
import pandas as pd


def load_and_clean_data(path, fallback_url="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"):
    if not os.path.exists(path):
        print(f"Local dataset not found at {path}. Fetching from {fallback_url}...")
        path = fallback_url
        
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df
