# Business Metric–Driven Churn Intervention System

## Overview

Customer churn prediction by itself rarely creates business value. Retention actions such as discounts, loyalty offers, or customer support interventions come with real costs, and blindly targeting every high-risk customer often leads to wasted spend.

This project reframes churn modeling as a **business decision problem**, not just a classification task.  
Instead of asking *“Who will churn?”*, the system answers:

> **“Which customers should the business actively intervene on to maximize net revenue?”**

The solution combines churn probability, customer lifetime value (CLV), and intervention cost to produce **actionable, cost-aware retention decisions**.

---

## Problem Statement

Most churn projects stop at predicting whether a customer will leave. In practice, this approach has two major flaws:

1. **Retention actions are not free** – offers, discounts, and service calls have costs.
2. **Not all customers are equally valuable** – saving a low-value customer may cost more than it returns.

The business problem addressed in this project is:

> *How can we identify customers who are both likely to churn and financially worth saving, while avoiding unnecessary retention costs?*

---

## Solution Design

The system follows a production-inspired design with a clear separation between **model training**, **inference**, and **business decisioning**.

### High-Level Architecture

Offline Training
└── train_model.py
↓
churn_model.pkl (saved model)

Batch Inference & Decisioning
└── main.py
↓
churn probabilities
↓
CLV + cost-based business logic
↓
Retention decisions & impact analysis

This structure mirrors how churn systems are typically implemented in real organizations.

---

## Dataset

The project uses the **IBM Telco Customer Churn dataset**, which contains:

- 7,032 customer records
- Demographic information
- Service usage and contract details
- Billing information (`MonthlyCharges`, `TotalCharges`)
- Target variable: `Churn` (Yes / No)

The dataset is well-suited for both churn prediction and value-based analysis.

---

## Modeling Approach

### Model Choice

A **Logistic Regression** model is used for churn prediction.

This choice is intentional:

- The project relies on **probability estimates**, not just class labels.
- Logistic Regression provides stable and reasonably well-calibrated probabilities.
- Interpretability is important for business-facing decisions.
- The focus is on **decision quality**, not marginal gains in accuracy.

The model is trained **offline** and saved as a reusable artifact.

---

## Business Logic

### Customer Lifetime Value (CLV)

A conservative CLV approximation is used to avoid overstating business impact:

CLV = MonthlyCharges × RemainingMonths × ContributionMargin

- RemainingMonths = (average tenure − current tenure), with a minimum of 1
- Contribution margin is set to 30% to reflect realistic profit, not raw revenue

### Expected Net Gain

For each customer:

ExpectedRevenueSaved = P(churn) × CLV
NetGain = ExpectedRevenueSaved − InterventionCost

This formulation allows the system to explicitly compare **expected benefit vs cost**.

---

## Customer Segmentation

Based on churn probability and net gain, customers are segmented into:

| Segment | Description |
|------|-------------|
| **Saveable** | High churn risk and positive expected net gain |
| **Loyal** | Low churn risk; no intervention required |
| **Not Worth Saving** | High churn risk but negative net gain |

Only customers in the **Saveable** segment are recommended for retention actions.

---

## Results Summary

When applied to the full customer base, the system identifies a clear prioritization
strategy for retention efforts:

- Approximately **20%** of customers are classified as **Saveable**, indicating high
  churn risk with positive expected net value from intervention
- Roughly **75–80%** of customers are categorized as **Loyal**, requiring no retention
  action and allowing the business to avoid unnecessary spend
- A small remainder is identified as **Not Worth Saving**, where intervention costs
  exceed expected value

This targeted segmentation enables focused retention strategies rather than blanket
campaigns, improving efficiency and aligning spend with measurable business impact.

### Sensitivity Analysis

The decision framework was evaluated across multiple intervention cost scenarios
(e.g., 20, 50, and 100 units).  
Across this range, the system consistently maintains positive expected value, indicating
that retention decisions are robust to reasonable variations in cost assumptions.

---

## Project Structure

CustomerChurnML/
├── data/
│ └── Customer Churn.csv
├── models/
│ └── churn_model.pkl
├── data_processing.py
├── train_model.py
├── model_inference.py
├── business_logic.py
├── evaluation.py
└── main.py

---

## How to Run

### 1. Train the model (one-time step)

```bash
python CustomerChurnML/train_model.py
```
This trains the churn model and saves it to models/churn_model.pkl.

### 2. Run inference and business decision

```bash
python CustomerChurnML/main.py
```
This executes batch inference, applies business logic, and prints:
- Customer segmentation counts 
- Estimated net business impact 
- Sensitivity analysis results 

## Key Takeaways
- Churn prediction alone is insufficient for effective retention strategies
- Business decisions should be driven by expected value, not accuracy metric
- Simple, interpretable models combined with strong business logic can outperform complex but misaligned approaches
- Cost-aware decision frameworks are critical for real-world ML systems

## Future Improvements
- Persist preprocessing artifacts (scalers, encoders) alongside the model
- Incorporate uplift modeling to estimate treatment effect of interventions
- Add A/B testing framework for retention strategies
- Monitor data drift and model performance over time