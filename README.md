# ☀️ Solar GHI Day-Ahead Forecasting

This project develops a **machine learning framework** for forecasting **Global Horizontal Irradiance (GHI)** using meteorological and temporal features. Accurate solar irradiance forecasting is critical for renewable energy planning and efficient grid integration.  

The final **XGBoost model** achieved a **Test MAPE of 14.53%** (excluding near-zero nighttime values), demonstrating strong predictive performance and robustness across weather conditions.

---

## 📌 Project Overview
- Forecast **solar irradiance (GHI)** values one day ahead.
- Dataset: historical solar radiation & weather features.
- Target: **hourly GHI**.
- Applications: **solar energy scheduling, smart grids, load balancing**.

---

## 🛠️ Data Preprocessing & Feature Engineering
- **Data Cleaning**
  - Removed features with >65% missing values.
  - Eliminated highly correlated features (correlation > 0.90).
  - Imputed missing values using median imputation.
  - Applied **IQR-based outlier capping**.

- **Feature Engineering**
  - Temporal: Hour, Day, Month, Day-of-week, Part-of-day (dawn/morning/afternoon/evening/night).
  - Lag features: `GHI lag-1`, `GHI diff-1` (temporal dependencies).
  - Categorical encoding for part-of-day.

---

## 🔍 Exploratory Data Analysis

### Target Variable Distribution
- GHI shows a **right-skewed distribution**: zero values at night, peaks during midday.  

![GHI Distribution](images/ghi_distribution.png)

### Temporal Patterns
- Clear **diurnal** and **weekly** patterns.  
- Highest irradiance in **morning & afternoon**.  

![GHI Temporal Patterns](images/ghi_temporal.png)

### Feature Relationships
- Key predictive features:
  - **Hour of day** (strong diurnal effect).
  - **Ambient temperature** (positive correlation).
  - **Lagged GHI** (sequential dependency).  

![Correlation Heatmap](images/ghi_correlation.png)

---

## 🤖 Model Architecture
Implemented and compared three models:

1. **Linear Regression** – simple baseline, poor generalization.  
2. **Random Forest Regressor** – strong baseline, robust non-linear modeling.  
3. **XGBoost Regressor** – best performer, capturing non-linear temporal & weather dependencies.  

### Configuration
- Train/Validation/Test = **60/20/20 split**  
- Scaling: **StandardScaler**  
- Hyperparameter Tuning: **GridSearchCV (3-fold CV)**  

---

## ⚙️ Hyperparameter Tuning
- **Random Forest:** `n_estimators=200`, `max_depth=20`  
- **XGBoost:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=6`  

---

## 📊 Evaluation Method
- **Custom MAPE** (ignores near-zero nighttime values):  
  \[
  \text{MAPE} = \frac{1}{n}\sum_{i \in S} \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100, \quad S = \{i: y_i \notin (-1,1)\}
  \]
- Additional metrics:
  - **RMSE**
  - **MAE**
  - **R²**

---

## 🏆 Results

| Model             | Validation MAPE (%) | Test MAPE (%) | Status        |
|-------------------|----------------------|---------------|---------------|
| Linear Regression | ~0 (train overfit)   | 1348.56       | ❌ Overfitting |
| Random Forest     | 18.99                | 19.20         | ✅ Good        |
| **XGBoost**       | **16.66**            | **14.53**     | 🏆 Excellent   |

### Time Series Performance
XGBoost predictions closely follow actual GHI patterns across time.  

![Actual vs Predicted GHI](images/ghi_actual_vs_pred.png)

---

## 🔑 Key Insights
- **Most important features:**
  - Hour, Part-of-day, GHI lag-1, Ambient temperature.
- **Model behavior:**
  - XGBoost best captured **non-linear temporal dynamics**.
  - Random Forest provided a solid baseline.
  - Linear Regression underperformed due to oversimplification.

---

## ⚡ Challenges
- Handling **missing & highly correlated features**.
- Adjusting evaluation for **zero nighttime GHI values**.
- Computational cost of **Grid Search hyperparameter tuning**.

---

## ✅ Conclusion
- Final **XGBoost model** achieved **14.53% Test MAPE**.
- Robust performance across different weather conditions.
- Framework can be extended for:
  - Real-time solar energy forecasting
  - Integration with **energy storage optimization**
  - Broader **renewable energy management systems**

---

## 🚀 Tech Stack
- Python (pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn)
- Jupyter Notebook
- Custom evaluation metrics

---

## 📦 Usage
```bash
# Clone repository
git clone https://github.com/your-username/solar-ghi-forecasting.git
cd solar-ghi-forecasting

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook assessment.ipynb
