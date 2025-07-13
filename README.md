# 🟡 Golden Moves: Forecasting Gold Price Trends with Stacking ML

## 🧠 Project Overview

Gold is considered a safe-haven asset, making its price prediction crucial for investors, analysts, and financial strategists. This project aims to build a robust multi-class classification model that predicts the **next-day price movement of gold (GLD)** — whether it will **rise**, **fall**, or **remain stable**.

Using technical indicators, feature selection, class balancing, and a powerful stacking ensemble classifier, we aim to achieve strong predictive performance.

---

## 📊 Feature Engineering Highlights

We created multiple features based on domain knowledge from financial time series analysis:

| Feature            | Description |
|--------------------|-------------|
| `Price_Change`     | % change between today and tomorrow |
| `GLD_lag1, lag2`   | Previous day(s) price |
| `GLD_MA3, MA7`     | Moving averages |
| `GLD_LogRet`       | Logarithmic return |
| `GLD_Volatility`   | Rolling 5-day standard deviation |
| `RSI`              | 14-day Relative Strength Index |
| `MACD`, `MACD_signal` | Momentum indicators |
| `Bollinger_upper/lower` | 20-day bands ± 2 std |

Target classes:
- `0`: Down (< -1%)
- `1`: Stable (between -1% and +1%)
- `2`: Up (> +1%)

---

## 🛠️ Preprocessing Steps

- Convert date column to `datetime`
- Feature selection with `ExtraTreesClassifier`
- Standardization via `StandardScaler`
- Class balancing with `SMOTEENN`
- Train/Test split with stratified sampling

---

## 🔍 Feature Selection

Using `ExtraTreesClassifier` + `SelectFromModel`, we retained features with highest importance to reduce overfitting and improve training speed.

---

## 🎯 Hyperparameter Tuning

Used **Optuna** to tune `XGBoost`:
- Objective: maximize **weighted F1 score**
- Tuned parameters: `n_estimators`, `max_depth`, `learning_rate`, `lambda`, `alpha`

---

## ♟️ Stacking Ensemble Classifier

Combined multiple models:
- **Base Models**: `XGBoost`, `LightGBM`, `CatBoost`
- **Meta Learner**: `Logistic Regression`

This ensemble improves performance and generalization.

---

## 📈 Evaluation Metrics

- **Accuracy**
- **Weighted F1 Score**
- **Precision**
- **Cohen’s Kappa**
- **Classification Report**
- **Confusion Matrix**

📉 Also includes True vs Predicted class plot for first 100 test points.

---

## 🚀 Inference Example

Simulate prediction on new (hypothetical) data with the pipeline:
- Transforms features
- Predicts next-day movement (`0`, `1`, or `2`)

---

## 🧾 Final Conclusion

✅ Full ML pipeline for 3-class financial classification  
✅ Strong feature engineering with technical indicators  
✅ Robust modeling with stacking ensemble  
✅ Well-documented evaluation

> Although originally treated as a regression problem, I reformulated it as classification to enable more actionable predictions in real-world financial strategy.

---

## 🔗 Explore More

📌 Kaggle Notebook: [Golden Moves on Kaggle](https://www.kaggle.com/code/saalnouramani/gold-price-movement-prediction)  
