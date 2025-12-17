from data_processing import load_and_clean_data
from model_inference import load_model_and_predict
from business_logic import churn_intervention_decision
from evaluation import business_summary


def main():
    df = load_and_clean_data("data/Customer Churn.csv")

    churn_probs = load_model_and_predict(df)

    decision_df = churn_intervention_decision(
        df,
        churn_probs,
        intervention_cost=50,
        churn_threshold=0.5
    )

    summary = business_summary(decision_df)

    print("Business Impact Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nSensitivity Analysis:")
    for cost in [20, 50, 100]:
        df_decision = churn_intervention_decision(
            df,
            churn_probs,
            intervention_cost=cost
        )
        net_gain = df_decision[df_decision["Segment"] == "Saveable"]["NetGain"].sum()
        print(f"Cost {cost}: Net Gain = {round(net_gain, 2)}")


if __name__ == "__main__":
    main()
