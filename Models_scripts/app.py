import streamlit as st

from data_processing import load_and_clean_data
from model_inference import load_model_and_predict
from business_logic import churn_intervention_decision
from evaluation import business_summary


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Churn Intervention Decision System",
    layout="wide"
)

st.title("Churn Intervention Decision System")
st.markdown(
    """
This dashboard supports **business-driven retention decisions** by identifying
customers who are likely to churn *and* financially worth saving.

The system combines churn probability, customer lifetime value (CLV),
and intervention cost to estimate expected net business impact.
"""
)

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Decision Parameters")

intervention_cost = st.sidebar.slider(
    "Intervention Cost per Customer",
    min_value=10,
    max_value=150,
    value=50,
    step=10
)

churn_threshold = st.sidebar.slider(
    "Churn Risk Threshold",
    min_value=0.3,
    max_value=0.7,
    value=0.5,
    step=0.05
)

st.sidebar.markdown(
    """
**Notes**
- Higher intervention cost reduces the number of customers worth targeting  
- Churn threshold controls how aggressively the business intervenes
"""
)

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_data():
    return load_and_clean_data("data/Customer Churn.csv")


df = load_data()

# --------------------------------------------------
# Run inference
# --------------------------------------------------
churn_probs = load_model_and_predict(df)

# --------------------------------------------------
# Business decisioning
# --------------------------------------------------
decision_df = churn_intervention_decision(
    df,
    churn_probs,
    intervention_cost=intervention_cost,
    churn_threshold=churn_threshold
)

summary = business_summary(decision_df)

# --------------------------------------------------
# KPI section
# --------------------------------------------------
st.subheader("Business Impact Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", summary["Total Customers"])
col2.metric("Saveable Customers", summary["Saveable Customers"])
col3.metric("Loyal Customers", summary["Loyal Customers"])
col4.metric("Expected Net Gain", f"{summary['Total Expected Net Gain']:.2f}")

# --------------------------------------------------
# Segmentation visualization
# --------------------------------------------------
st.subheader("Customer Segmentation Distribution")

segment_counts = decision_df["Segment"].value_counts()
st.bar_chart(segment_counts)

# --------------------------------------------------
# Decision table (optional view)
# --------------------------------------------------
with st.expander("View Sample Retention Decisions"):
    st.dataframe(
        decision_df[
            [
                "customerID",
                "P_churn",
                "CLV",
                "ExpectedRevenueSaved",
                "NetGain",
                "Segment",
            ]
        ].head(20)
    )

# --------------------------------------------------
# Download decision CSV
# --------------------------------------------------
st.subheader("Download Retention Decisions")

csv_data = decision_df.to_csv(index=False)

st.download_button(
    label="Download Decision CSV",
    data=csv_data,
    file_name="churn_intervention_decisions.csv",
    mime="text/csv"
)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "This dashboard is a lightweight interface for a business-driven churn "
    "intervention system. Model training is performed offline; the UI focuses "
    "on inference and decision analysis."
)