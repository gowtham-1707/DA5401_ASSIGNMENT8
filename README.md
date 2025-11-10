# DA5401 A8 — Ensemble Learning for Complex Regression Modeling on Bike Share Data

## Project Overview
This project applies **ensemble learning techniques** — **Bagging**, **Boosting**, and **Stacking** — to a **bike-sharing demand prediction** problem using the *UCI Bike Sharing Dataset* (hourly data).  
The objective is to build accurate regression models that forecast hourly bike rentals (`cnt`) based on environmental, temporal, and seasonal factors.

---

## Dataset Description
**Source:**  
Hadi Fanaee-T & João Gama, *Progress in Artificial Intelligence (2013)*  
> [UCI Machine Learning Repository – Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)

**File used:** `hour.csv`  
**Records:** 17,379 (hourly data for 2011–2012)  
**Target Variable:** `cnt` (total bike rentals = `casual + registered`)

### Feature Overview
| Type | Columns | Description |
|------|----------|--------------|
| Categorical | `season`, `weathersit`, `mnth`, `hr`, `weekday`, `workingday`, `holiday` | Encoded as dummy variables |
| Numerical | `temp`, `atemp`, `hum`, `windspeed` | Scaled using StandardScaler |
| Dropped | `instant`, `dteday`, `casual`, `registered` | Irrelevant for prediction |
| Target | `cnt` | Total rentals (output variable) |

---

##  Steps and Methodology

### 1️ Data Preprocessing
- Loaded `hour.csv` using pandas.
- Dropped irrelevant columns (`instant`, `dteday`, `casual`, `registered`).
- One-hot encoded categorical variables.
- Scaled numeric variables using `StandardScaler`.
- Applied **time-based 80/20 split** (training: 13,903 rows, testing: 3,476 rows).

### 2 Baseline Models
| Model | Purpose | Notes |
|--------|----------|-------|
| **Decision Tree (max_depth=6)** | Non-linear baseline | Captures simple rules |
| **Linear Regression** | Linear baseline | Provides RMSE benchmark |

The better RMSE between these two is used as the **baseline performance metric**.

### 3 Ensemble Techniques
| Technique | Implementation | Focus | Key Idea |
|------------|----------------|--------|-----------|
| **Bagging** | `BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=6))` | Variance Reduction | Averages many Decision Trees |
| **Boosting** | `GradientBoostingRegressor()` | Bias Reduction | Sequentially fits residuals |
| **Stacking** | `StackingRegressor([KNN, Bagging, Boosting], final_estimator=Ridge)` | Combines both | Meta-learning from diverse models |

---

## Evaluation Metric
**Root Mean Squared Error (RMSE)**  
\[
RMSE = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}(y_i - \\hat{y}_i)^2}
\]

RMSE is used because it penalizes large errors and is suitable for regression forecasting.

---

## Results Summary

| Model | RMSE (↓ Better) | Interpretation |
|--------|----------------:|----------------|
| Decision Tree | 159.13 | High variance, moderate bias |
| Linear Regression | 133.83 | Lower bias, linear trend captured |
| Bagging | *156.20* | Reduced variance vs. Decision Tree |
| Gradient Boosting | *101.84* | Reduced bias vs. Bagging |
| Stacking | *101.45* | Best overall — combines diversity |


---

## Interpretation of Results
- **Bagging** successfully reduced variance, stabilizing Decision Tree predictions.  
- **Boosting** further reduced bias, modeling complex temporal-weather interactions.  
- **Stacking** achieved the **lowest RMSE**, confirming that blending diverse learners yields superior generalization.

This aligns with ensemble learning theory — diverse models cooperate to balance **bias–variance trade-off**, leading to optimal predictive performance.

---

## Conclusion
> The **Stacking Regressor** outperformed all other models, effectively combining KNN, Bagging, and Boosting regressors through a Ridge meta-learner.  
> This ensemble approach demonstrates that hybrid learning methods can enhance regression accuracy and reliability, making them ideal for dynamic forecasting tasks like bike rental demand prediction.

---

## File Structure
