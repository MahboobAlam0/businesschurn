import numpy as np

print(">>> USING UPDATED business_logic.py WITH MARGIN <<<")
def compute_clv(df, margin=0.3):
    avg_tenure = df["tenure"].mean()
    remaining_months = np.maximum(1, avg_tenure - df["tenure"])
    gross_value = df["MonthlyCharges"] * remaining_months
    return gross_value * margin




def churn_intervention_decision(
    df,
    churn_probs,
    intervention_cost=50,
    churn_threshold=0.4
):
    df = df.copy()
    df["P_churn"] = churn_probs
    df["CLV"] = compute_clv(df)

    df["ExpectedRevenueSaved"] = df["P_churn"] * df["CLV"]
    df["NetGain"] = df["ExpectedRevenueSaved"] - intervention_cost

    def segment(row):
        if row["P_churn"] < churn_threshold:
            return "Loyal"
        elif row["NetGain"] > 0:
            return "Saveable"
        else:
            return "Not Worth Saving"

    df["Segment"] = df.apply(segment, axis=1)

    return df
