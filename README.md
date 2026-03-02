# Business Metric–Driven Churn Intervention System

## Overview

Predicting customer churn alone does **not** create business value.

Retention actions such as discounts, loyalty offers, or customer support outreach carry **real operational costs**, and targeting every high-risk customer often results in wasted spend.

This project reframes churn modeling as a **business decision problem**, not just a classification task.

Instead of asking:

> *“Which customers are likely to churn?”*

the system answers:

> **“Which customers should we intervene on to maximize expected net revenue?”**

The solution combines **churn probability**, **customer lifetime value (CLV)**, and **intervention cost** to produce **cost-aware, actionable retention decisions**.

---

## Problem Framing

Most churn projects stop at predicting a binary outcome (churn / no churn).  
This approach breaks down in practice because:

1. **Retention actions are not free** — discounts and support interactions have costs.
2. **Customers are not equally valuable** — saving a low-value customer may cost more than it returns.
3. **Probabilities matter more than labels** — decision-making requires risk estimates, not hard predictions.

The business question addressed here is:

> *How can we identify customers who are both likely to churn **and** financially worth saving, while avoiding unnecessary retention costs?*

---

## System Design

The project follows a **production-inspired structure** with a clear separation between:

- **Model training**
- **Batch inference**
- **Business decision logic**

### High-Level Architecture

```
Offline Training
└── train_model.py
↓
model.pkl (saved model)

Batch Inference & Decisioning
└── main.py
↓
churn probabilities
↓
CLV + cost-aware business logic
↓
Retention decisions & impact analysis
```


This mirrors how churn pipelines are commonly implemented in real organizations:
- models are trained offline,
- predictions are generated in batches,
- business logic determines actions.

---

## Dataset

The system uses the **IBM Telco Customer Churn Dataset**, containing:

- **7,032 customer records**
- Demographic and service usage features
- Contract and billing information
- Target variable: `Churn` (Yes / No)

The dataset supports both churn prediction and value-based segmentation.

---

## Modeling Approach

### Model Choice

A **Logistic Regression** model is used for churn prediction.

This is a **deliberate design decision**:

- The system relies on **probability estimates**, not class labels.
- Logistic Regression provides **stable and interpretable probabilities**.
- Calibration and reliability are more important than marginal accuracy gains.
- The focus is on **decision quality**, not leaderboard metrics.

The model is trained offline and saved as a reusable artifact.

---

## Model Performance

Evaluation on a held-out test set:

| Metric | Value |
|------|------|
| Accuracy | **0.805** |
| ROC-AUC | **0.836** |

These results are **sufficient for decision support**, where business logic absorbs model uncertainty.

---

## Business Logic

### Customer Lifetime Value (CLV)

A conservative CLV approximation is used to avoid overstating impact:
**CLV = MonthlyCharges × RemainingMonths × ContributionMargin**


Assumptions:
- RemainingMonths = (average tenure − current tenure), minimum of 1
- Contribution margin fixed at **30%** to reflect profit, not revenue

---

### Expected Net Value

For each customer:
**Expected Revenue Saved = P(churn) × CLV**
**Net Gain = Expected Revenue Saved - Intervention Cost**


This formulation explicitly compares **expected benefit vs cost**.

---

## Customer Segmentation

Customers are segmented based on churn probability and expected net gain:

| Segment | Description |
|------|-------------|
| **Saveable** | High churn risk and positive expected net gain |
| **Loyal** | Low churn risk; no intervention required |
| **Not Worth Saving** | High churn risk but negative net gain |

Only **Saveable** customers are recommended for retention actions.

---

## Results: Business Impact

### Overall Summary

| Metric | Value |
|------|------|
| Total Customers | 7,032 |
| Saveable Customers | **1,424 (~20%)** |
| Loyal Customers | **5,455 (~78%)** |
| Not Worth Saving | **153 (~2%)** |
| Total Expected Net Gain | **≈ 449,500** |

This segmentation enables **targeted retention efforts** instead of blanket campaigns.

---

## Sensitivity Analysis

The decision framework was evaluated across multiple intervention cost assumptions:

| Intervention Cost | Expected Net Gain |
|------|------|
| 20 | **≈ 553,884** |
| 50 | **≈ 499,251** |
| 100 | **≈ 411,593** |

Key observation:
- Net gain decreases as cost increases (expected behavior)
- The strategy remains **profitable across all tested scenarios**

This indicates that the retention policy is **robust**, not brittle.

---

## Project Structure

```
businesschurn/
├── data/
│ └── CustomerChurn.csv
├── models/
│ └── churn_model.pkl
├── Models_Scripts/
│ ├── app.py
│ ├── business_logic.py
│ ├── data_processing.py
│ ├── evaluation.py
│ ├── train_model.py
│ ├── model_inference.py
│ └── main.py
├── README.md
└── requirements.txt
```

---

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python Models_Scripts/train_model.py
```

3. Run the application:
```bash
python Models_Scripts/app.py
```

Outputs:

- Customer segmentation counts
- Total expected net business gain
- Sensitivity analysis across intervention costs

## Key Takeaways

- Churn prediction alone is insufficient for retention strategy design
- Business decisions should be driven by expected value, not accuracy
- Simple, interpretable models paired with strong business logic are often superior to complex models without cost awareness
- Probability-based decision systems are more realistic than binary classification pipelines

## Limitations
- CLV is approximated using simple heuristics rather than a dedicated CLV model
- No uplift modeling — treatment effect of interventions is assumed, not estimated
- No live A/B testing loop
- No long-term monitoring or drift detection

These limitations are explicitly documented to avoid overstating conclusions.

## Future Improvements
- Persist preprocessing artifacts alongside the model
- Introduce uplift modeling for causal intervention estimation
- Add A/B testing simulation for retention strategies
- Implement data drift and performance monitoring
- Extend to multi-period retention optimization

## Note

This project is intended for educational and portfolio demonstration purposes only.
It is not production-deployed and does not represent a fully operational retention system.