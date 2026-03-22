# PJME Hourly Energy Demand Forecasting
> Hourly energy demand forecasting on the PJM East dataset
> using XGBoost with time series cross-validation.

---

## Results

<img width="833" height="453" alt="image" src="https://github.com/user-attachments/assets/2fc011ae-d4ac-4bac-9e1a-a36514ca72cf" />

Best model (V7 — Optuna Tuned) achieves **RMSE of 1789.8 MW** and **MAPE of 4.1%** on held-out validation sets using 5-fold time series cross-validation.

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

An anomalous trough was identified in 2013. After investigation, I decided to drop all rows below 19,000 MW as they represent sensor or recording errors.

---

## Feature Engineering

**Time features**

Capture cyclical patterns in demand.

| Feature | Description |
|---------|-------------|
| `hour` | Hour of day (0–23) |
| `dayofweek` | Day of week (0=Mon, 6=Sun) |
| `month` | Month of year |
| `year` | Calendar year — captures long-term demand trend |
| `dayofyear` | Day of year (1–365) |
| `weekofyear` | Week number of year |

**Annual lag features**

Same-hour demand values from 1, 2, and 3 years prior. A 364-day offset is used instead of 365 to preserve day-of-week alignment.

| Feature | Offset |
|---------|--------|
| `lag1` | 364 days |
| `lag2` | 728 days |
| `lag3` | 1092 days |

**Rolling features**

Capture short-term demand trends. Computed with a 1-step shift to prevent data leakage.

| Feature | Window |
|---------|--------|
| `rolling_mean_24h` | 24-hour rolling mean |
| `rolling_mean_168h` | 168-hour (1 week) rolling mean |

**Holiday feature**

| Feature | Description |
|---------|-------------|
| `is_holiday` | Binary flag for US federal holidays |

---

## Model Iterations: v1 to v7

All versions use 5-fold `TimeSeriesSplit` with `test_size=24×7×12` (12 weeks per fold) and `gap=24` hours between train and validation to prevent leakage. XGBoost hyperparameters are held constant across V1–V6 to isolate the effect of feature changes, then tuned in V7.

| Model | Key Changes | RMSE | MAPE | Δ RMSE |
|-------|-------------|------|------|--------|
| V1 — Baseline | 6 time features (`hour`, `dayofweek`, `month`, `year`, `dayofyear`, `weekofyear`) | 3831.9 MW | 9.0% | — |
| V2 — Annual lags | Added `lag1`, `lag2`, `lag3` | 3855.0 MW | 9.0% | +23.1 |
| V3 — Rolling features | Added `rolling_mean_24h`, `rolling_mean_168h` | 1883.0 MW | 4.3% | -1972.0 |
| V4 — Short-term lags | Added `lag_24h`, `lag_48h`, `lag_168h` | 1890.4 MW | 4.4% | +7.4 |
| V5 — More signal features | Added `is_holiday`, `is_weekend`, `is_peak_hour` | 1890.3 MW | 4.4% | -0.1 |
| V6 — Cleaned feature set | Removed redundant features (`is_weekend`, `is_peak_hour`, short-term lags) | 1885.3 MW | 4.4% | -5.0 |
| **V7 — Optuna** | Bayesian hyperparameter search (50 trials) | **1799.5 MW** | **4.1%** | **-85.8** |

The rolling features in V3 produced the largest drop of 1,972 MW RMSE, achieved by giving the model direct access to recent demand levels rather than relying solely on same-period values from prior years.

V2 adding annual lags slightly worsened RMSE before rolling features were added, as early folds lacked sufficient history for the lag lookups, returning NaN for a portion of training rows.

---

## Hyperparameter Tuning

V7 used [Optuna](https://optuna.org/) Bayesian optimisation over 50 trials to find the best XGBoost configuration.

**Best parameters found:**

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 0.0292 |
| `max_depth` | 8 |
| `subsample` | 0.715 |
| `colsample_bytree` | 0.839 |
| `min_child_weight` | 9 |
| `reg_alpha` | 0.635 |
| `reg_lambda` | 2.132 |

The higher `min_child_weight` and `reg_alpha` indicate the optimal model is more regularised than the manually tuned baseline. The faster `learning_rate` of 0.029 (vs 0.01) meant early stopping triggered at 432 trees rather than ~1993, making V7 significantly faster to train.

---

## Error Analysis

**Feature importances (V7, averaged across 5 folds)**

<img width="832" height="389" alt="image" src="https://github.com/user-attachments/assets/8bfdefbe-b3e0-4c0c-af1f-a35c4417eefe" />

`lag1`, `lag3`, `rolling_mean_24h`, and `lag2` dominate feature importances. `hour` remains a meaningful contributor. The remaining time features contribute minimally as their signal is largely captured by the lag and rolling features.

---

## Potential Improvements

- **Temperature data** — weather drives demand spikes that no lag or calendar feature can anticipate
- **LightGBM benchmark** — faster training with comparable accuracy, should be quite straightforward to plug into the existing experiment framework
- **Broader Optuna search** — increase to 100+ trials and widen the search space for `learning_rate` and `max_depth`

---

## Requirements
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna
```
