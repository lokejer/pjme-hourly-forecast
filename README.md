# PJME Hourly Energy Demand Forecasting
> Hourly energy demand forecasting on the PJM East dataset
> using XGBoost with time series cross-validation.

---

## Results

<img width="1253" height="451" alt="Screenshot 2026-03-20 004644" src="https://github.com/user-attachments/assets/fcd824ff-ada1-477d-9853-9035e1d03406" />

Best model (v4) achieves **RMSE of 1861.9 MW** on a held-out validation set using 5-fold time series cross-validation.

---

## Dataset

The dataset contains hourly electricity demand (in MW) for the PJM East region from 2002 to 2018, sourced from [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).

| Property | Value |
|----------|-------|
| Source | PJM Interconnection LLC |
| Period | 2002 – 2018 |
| Frequency | Hourly |
| Rows | ~145,000 |
| Target | `PJME_MW` |

I identified an anomalous trough in 2013, so I dropped all rows below 19,000MW which likely represented sensor/recording errors.

---

## Feature Engineering

Features were built in three stages, each corresponding to a model iteration.

**Time features**

Capture cyclical patterns in demand.

| Feature | Description |
|---------|-------------|
| `hour` | Hour of day (0–23) |
| `dayofweek` | Day of week (0=Mon, 6=Sun) |
| `month` | Month of year |
| `quarter` | Quarter of year |
| `year` | Calendar year |
| `dayofyear` | Day of year (1–365) |
| `dayofmonth` | Day of month |
| `weekofyear` | Week number of year |

**Lag features**

Capture yearly shifts in demand due to various seasonal factors like summertime heat.
Lag features are same-hour values from 1, 2, and 3 years prior. A 364-day offset is used so the lag lands on the same day of the week.

| Feature | Offset |
|---------|--------|
| `lag1` | 364 days |
| `lag2` | 728 days |
| `lag3` | 1092 days |

**Rolling features**

Capture short-term demand trends.
The rolling features were computed using a 1-step shift to prevent data leakage.

| Feature | Window |
|---------|--------|
| `rolling_mean_24h` | 24-hour rolling mean |
| `rolling_mean_168h` | 168-hour (1 week) rolling mean |

---

## Model Iterations

5-fold `TimeSeriesSplit` with `test_size=24×7×12` (12weeks per fold) and a `gap=24` hours between train and validation to prevent leakage.

| Model | Key Changes | RMSE | Δ RMSE |
|-------|-------------|------|--------|
| v1 Baseline | 8 time features (`hour`, `dayofweek`, `month`, `quarter`, `year`, `dayofyear`, `dayofmonth`, `weekofyear`) | 3959.3 MW | — |
| v2 + Lag Features | Added `lag1`, `lag2`, `lag3` (same hour 1/2/3 years prior) | 3836.5 MW | -122.8 |
| v3 + Rolling Features | Added `rolling_mean_24h`, `rolling_mean_168h` | 2285.4 MW | -1551.1 |
| **v4 (Best)** | `n_estimators=2000`, `max_depth=5`, `subsample=0.8`, `colsample_bytree=0.8` | **1861.9 MW** | **-423.5** |

The rolling features in v3 produced the largest single improvement, a drop of 1,551 MW in RMSE, by giving the model direct access to recent demand levels rather than relying solely on same-period values from prior years.

---

## Error Analysis

**Feature importances (v4, averaged across 5 folds)**

<img width="1206" height="724" alt="image" src="https://github.com/user-attachments/assets/e5687fca-5211-4c01-8760-b612e8908af2" />

`lag1`, `rolling_mean_24h`, `lag2`, and `lag3` dominate feature importances. `hour` remains a meaningful contributor. The remaining time features (`dayofyear`, `weekofyear`, etc.) contribute minimally, likely encoding residual seasonality that the lag and rolling features already capture.

---

## Potential Improvements

- **Hyperparameter tuning** — systematic search using Optuna (Bayesian optimisation) across `learning_rate`, `max_depth`, `subsample`, and regularisation parameters
- **LightGBM benchmark** — faster training with comparable accuracy; worth comparing against v4 directly
- **Additional features** — `is_weekend`, `is_peak_hour` (08:00–21:00), public holiday flags
- **Baseline Comparison** — compare with linear models and RandomForest regressor
