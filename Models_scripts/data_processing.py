import pandas as pd


def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Fix TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop rows with missing TotalCharges
    df = df.dropna(subset=["TotalCharges"])

    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df
