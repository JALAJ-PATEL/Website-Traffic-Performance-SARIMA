# 📈 Website Traffic Forecasting using SARIMA

This project explores time series forecasting using the **SARIMA (Seasonal AutoRegressive Integrated Moving Average)** model to predict website sessions from Google Analytics–exported data.

> **Repo Files**
- `data-export.csv` → Preprocessed website analytics data
- `Website-Traffic-Performance-SARIMA.ipynb` → Complete model pipeline + analysis
- `Website-Traffic-SARIMA.ipynb` → Final SARIMA modeling + evaluation + visualization

---

## 🔍 Problem Statement

Forecast the hourly website sessions to understand trends, cyclic behavior, and help with:
- Server load balancing
- Marketing campaign timing
- Business forecasting
- Anomaly detection

---

## 📦 Dataset

**Source**: `data-export.csv`

Columns include:
- `Datetime` (hourly timestamps)
- `Sessions` (website visits per hour)
- Channel grouping info: `"Direct"`, `"Organic Social"`, etc.
- Engagement metrics: average session time, bounce rate, event count, etc.

> We focused on the `Sessions` column for univariate forecasting.

---

## 🧠 Why SARIMA?

**SARIMA** extends ARIMA by incorporating seasonality (`S`). It models:
- `AR (AutoRegression)` → relationship between current value & previous values
- `I (Integration)` → differencing to make data stationary
- `MA (Moving Average)` → error of previous predictions
- `Seasonal components` → repeated patterns over time

Mathematically, SARIMA is written as:
Where:
- `p, d, q` → Non-seasonal orders
- `P, D, Q` → Seasonal orders
- `s` → Length of the seasonal cycle (e.g., 24 for daily seasonality in hourly data)

---

## 🧪 Modeling Pipeline (`Website-Traffic-Performance-SARIMA.ipynb`)

### ✅ Steps Followed:

1. **Data Preprocessing**
   - Parsed datetime
   - Filtered out recent hourly data
   - Converted `Sessions` to numeric
   - Resampled to hourly format (if needed)

2. **Stationarity Check**
   - Visual inspection (rolling mean & std)
   - ADF test for stationarity
   - Differencing applied as needed

3. **SARIMA Parameter Tuning**
   - Used **auto_arima** to select optimal `(p,d,q)(P,D,Q,s)`
   - Seasonal period `s = 24` (daily cycle in hourly data)

4. **Model Training**
   - Fit SARIMAX model on past data
   - Last 2 days of data used for training-validation split

5. **Forecasting**
   - Predicted `Sessions` for the next 7 days
   - Visualized actual vs predicted with confidence intervals

6. **Evaluation Metrics**
   - MAE: **26.06**
   - RMSE: **34.45**
   - MAPE: **12.30%**

7. **Plotting**
   - Forecast plot with pink shaded 95% CI
   - Training data vs actual vs predicted sessions

---

## 📘 Notebook Overview: `Website-Traffic-SARIMA.ipynb`

This notebook focuses on the **final SARIMA model** for performance evaluation:
- Uses the best-fit SARIMA parameters
- Forecasts multiple days ahead
- Visualizes results with:
  - Actual vs Predicted
  - Confidence bounds
- Summarizes model metrics for easy interpretation

It serves as the **production-ready version** of the model — useful for business dashboards or REST API integration for automated forecasting.

---

## 📚 Theoretical Significance of SARIMA

SARIMA is a **cornerstone of classical time series forecasting** and holds foundational value in:
- Statistical modeling
- Econometrics
- Signal processing

> Even with the rise of Deep Learning (LSTMs, Transformers), SARIMA remains relevant due to:
- **Interpretability** — clear components (trend, seasonality, noise)
- **Strong baseline** — often hard to beat without much more complex models
- **Data efficiency** — works well with small/medium datasets

Modern ML/DL time series models often **build upon the ideas** of SARIMA:
- Seasonality blocks in DeepAR
- Temporal convolution inspired by autoregression
- Trend-seasonality decomposition in N-BEATS, Prophet, etc.

---
