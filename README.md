# PJME Hourly Energy Demand Forecasting

Hourly electricity demand forecasting on the PJM East dataset using XGBoost with time series cross-validation and exogenous temperature variables.

---

## Business Question

**Can we accurately forecast hourly electricity demand for the PJM East region using historical consumption patterns and weather data?**

Electricity grid operators need accurate demand forecasts to balance supply and load in real time. Underestimating demand risks blackouts; overestimating it wastes fuel and money. This project builds a machine learning model that forecasts hourly electricity consumption in megawatts (MW), using historical demand patterns, calendar signals, and regional temperature data as inputs.

---

## Results

<img width="1198" height="411" alt="image" src="https://github.com/user-attachments/assets/50ca45e5-2b37-42cf-8f04-93c4765489db" />
> Best model (V8 — Optuna Tuned) achieves **RMSE of 1036.7 MW** and **MAPE of 2.4%** on a held-out validation set using 5-fold time series cross-validation — a **73% reduction in error from the baseline model**.

| Model | Key Changes | RMSE | MAPE | Δ RMSE |
|-------|-------------|------|------|--------|
| V1 — Baseline | Calendar features only | 3821.9 MW | 8.9% | — |
| V2 — Annual Lags | Added `lag1`, `lag2`, `lag3` | 3862.1 MW | 9.0% | +40.2 |
| V3 — Rolling Demand | Added `rolling_mean_24h`, `rolling_mean_168h` | 1885.2 MW | 4.3% | -1976.9 |
| V4 — Holiday | Added `is_holiday` flag | 1881.6 MW | 4.3% | -3.6 |
| V5 — Raw Temperature | Added `temperature` | 1169.0 MW | 2.8% | -712.6 |
| V6 — Heating/Cooling | Added `heating_degrees`, `cooling_degrees` | 1170.7 MW | 2.8% | +1.7 |
| V7 — Rolling Temperature | Added `rolling_temp_24h`, `rolling_temp_168h` | 1156.4 MW | 2.7% | -14.3 |
| **V8 — Optuna Tuned** | Bayesian hyperparameter optimisation | **1036.7 MW** | **2.4%** | **-119.7** |

---

## Dataset

The dataset contains hourly electricity demand (in MW) for the PJM East interconnection from 2002 to 2018, sourced from [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).

| Property | Value |
|----------|-------|
| Source | PJM Interconnection LLC |
| Period | 2002 – 2018 (processed data starts 2004-12-28 after lag feature creation) |
| Frequency | Hourly |
| Rows (after cleaning) | 119,139 |
| Target variable | `PJME_MW` (electricity demand in megawatts) |
| Demand range | 19,085 MW – 62,009 MW |
| Mean demand | 32,134 MW |

**Outlier removal:** An anomalous trough of readings below 19,000 MW was identified and removed. These represent recording errors rather than genuine demand — confirmed by plotting them, which showed isolated spikes inconsistent with normal seasonal patterns.

---

## Methodology

### What is XGBoost?

XGBoost (Extreme Gradient Boosting) is a tree-based machine learning algorithm that builds an ensemble of decision trees sequentially. Each new tree is trained to correct the errors (residuals) of the previous trees — this process is called **gradient boosting**.

**Why it suits time series forecasting:**
- Handles tabular feature inputs natively — calendar variables, lag values, and temperature features all map directly into the feature matrix
- Does not assume linear relationships, capturing the non-linear U-shaped relationship between temperature and demand
- Early stopping prevents overfitting by halting training when validation error stops improving
- Feature importance provides interpretability into which signals drive predictions

Unlike deep learning approaches (LSTMs, Transformers), XGBoost requires explicit feature engineering — the temporal structure must be encoded as features. This is a feature, not a limitation: it forces clear reasoning about what signals matter and why.

---

### Feature Engineering

All features are derived from the datetime index and historical demand values. No future data is used at any point.

#### Calendar Features

Derived directly from the datetime index to encode cyclical demand patterns.

| Feature | Signal |
|---------|--------|
| `hour` | Daily demand cycle — peaks at business/evening hours |
| `dayofweek` | Weekday vs weekend consumption patterns |
| `month` | Monthly seasonality |
| `quarter` | Broad seasonal grouping |
| `year` | Long-term demand trend across 2004–2018 |
| `dayofyear` | Precise position in the annual seasonal cycle |
| `dayofmonth` | Day within a month |
| `weekofyear` | Annual cycle at weekly resolution |

#### Annual Lag Features

The same hour from 1, 2, and 3 years prior. A **364-day offset** is used instead of 365 to preserve day-of-week alignment — 364 days = exactly 52 weeks, so `lag1` for a Tuesday always lands on a Tuesday.

| Feature | Offset | Signal |
|---------|--------|--------|
| `lag1` | 364 days | Same hour last year |
| `lag2` | 728 days | Same hour two years ago |
| `lag3` | 1,092 days | Same hour three years ago |

#### Rolling Demand Features

Short-term demand context computed with a `shift(1)` to prevent same-row data leakage.

| Feature | Window | Signal |
|---------|--------|--------|
| `rolling_mean_24h` | 24-hour rolling mean | Yesterday's average demand level |
| `rolling_mean_168h` | 168-hour rolling mean | Last week's average demand regime |

#### Calendar Event Features

| Feature | Description |
|---------|-------------|
| `is_holiday` | 1 if US federal holiday (via `USFederalHolidayCalendar`), 0 otherwise |
| `is_weekend` | 1 if Saturday or Sunday, 0 otherwise |
| `is_peak_hour` | 1 if hour is between 08:00–21:00, 0 otherwise |

#### Temperature Features (Exogenous)

Hourly temperature data was fetched from the [Open-Meteo Historical Archive API](https://archive-api.open-meteo.com) for four representative cities in the PJM East region: Philadelphia, Baltimore, Newark, and Wilmington. A simple unweighted average across the four cities was used as the regional temperature proxy.

| Feature | Description |
|---------|-------------|
| `temperature` | Average regional temperature in °C |
| `heating_degrees` | `max(0, 18.3 – temperature)` — cold stress above baseline |
| `cooling_degrees` | `max(0, temperature – 18.3)` — heat stress above baseline |
| `rolling_temp_24h` | 24-hour rolling mean temperature (shift(1)) |
| `rolling_temp_168h` | 168-hour rolling mean temperature (shift(1)) |

The **18.3°C (65°F) base temperature** is the industry-standard balance point used in US energy forecasting — below this threshold buildings require heating; above it they require cooling. Heating and cooling degree features encode the non-linear U-shaped relationship between temperature and electricity demand, which raw temperature alone does not cleanly represent.

---

### Validation Strategy

Standard k-fold cross-validation is invalid for time series — a model could train on 2015 data and test on 2010 data, learning the future to predict the past. This leaks information and produces unrealistically optimistic error estimates.

**`TimeSeriesSplit`** enforces strict temporal ordering: every training fold contains only data that precedes its test fold. This ensures the model is always evaluated on genuinely future data.

```
tscv = TimeSeriesSplit(
    n_splits=5,
    test_size=24 × 7 × 12,  # 12 weeks per fold
    gap=24                   # 24-hour gap between train and test
)
```

The `gap=24` parameter skips 24 hours between the last training point and the first test point, simulating a realistic 24-hour-ahead forecasting scenario and preventing adjacent-hour leakage.

Each fold's test window covers 2,016 hours (~12 weeks). The five fold windows are sequential, covering progressively later periods of the dataset.

**Evaluation metrics:**
- **RMSE** (Root Mean Squared Error) — primary metric, penalises large errors heavily, catches missed demand spikes
- **MAPE** (Mean Absolute Percentage Error) — percentage error, interpretable as "off by X% on average"

---

### Model Iterations

Each version adds one category of new signal, keeping XGBoost hyperparameters fixed (`n_estimators=2000`, `max_depth=5`, `learning_rate=0.01`) to isolate the contribution of features alone.

---

**V1 — Baseline (RMSE: 3821.9 MW, MAPE: 8.9%, 413 trees)**

Calendar features only. The model knows *when* it is but nothing about historical demand. It can approximate general daily and seasonal patterns but has no anchor to actual consumption levels. Predictions are consistently lower than actuals because the model has no demand history to reference.

---

**V2 — Annual Lags (RMSE: 3862.1 MW, MAPE: 9.0%, 399 trees)**

Added `lag1`, `lag2`, `lag3`. Counterintuitively, performance slightly worsened. The early CV folds do not yet have 364 days of history — those rows produce NaN lag values, which XGBoost handles via learned default directions that are not always optimal. The model also only used 399 trees (vs 413 in V1), indicating early stopping triggered sooner — the lags were not consistently helpful at this stage.

---

**V3 — Rolling Demand (RMSE: 1885.2 MW, MAPE: 4.3%, 1997 trees)**

The single largest improvement in the entire experiment — a drop of **1,976.9 MW RMSE**. Adding `rolling_mean_24h` and `rolling_mean_168h` gave the model direct access to recent demand levels. Instead of only knowing "it's 3pm on a Tuesday in July", the model now also knows "demand has been averaging 38,000 MW over the past 24 hours." This short-term context is far more informative than calendar features alone. The model used nearly all 2,000 trees, indicating the rolling features provided sustained learning signal.

---

**V4 — Holiday Flag (RMSE: 1881.6 MW, MAPE: 4.3%, 1999 trees)**

Marginal improvement (-3.6 MW). The `is_holiday` flag has limited incremental value because the annual lags already implicitly encode holiday patterns — `lag1` for Christmas 2016 looks up Christmas 2015, so the model already "knows" holiday demand is lower via the lag value itself.

---

**V5 — Raw Temperature (RMSE: 1169.0 MW, MAPE: 2.8%, 1996 trees)**

The second-largest single improvement — a drop of **712.6 MW RMSE**. Temperature is the primary driver of electricity demand spikes that no calendar or lag feature can anticipate. A cold snap or heatwave causes demand to surge in ways the model could previously only react to (via rolling demand) rather than anticipate. With raw temperature as a feature, the model can now see the cause, not just the effect.

---

**V6 — Heating/Cooling Degrees (RMSE: 1170.7 MW, MAPE: 2.8%, 1997 trees)**

No meaningful improvement over V5 (+1.7 MW). XGBoost can learn the non-linear U-shape relationship from raw temperature directly by splitting the feature space at the balance point threshold. The engineered degree features are redundant given the model's ability to discover this relationship itself. They are retained as features because `cooling_degrees` becomes the second-most important feature in V7/V8 — the model leverages them even if they did not independently move the RMSE at this step.

---

**V7 — Rolling Temperature (RMSE: 1156.4 MW, MAPE: 2.7%, 1999 trees)**

Small but consistent improvement (-14.3 MW). The 24-hour rolling temperature captures sustained heat or cold — a week where temperatures have not dropped below 28°C at night drives higher overnight demand than a single hot afternoon. `rolling_temp_168h` contributes near-zero importance and adds little beyond what raw temperature already provides at hourly resolution.

---

**V8 — Optuna Hyperparameter Tuning (RMSE: 1036.7 MW, MAPE: 2.4%, 1194 trees)**

Systematic Bayesian hyperparameter optimisation using [Optuna](https://optuna.org) over 50 trials, searching across `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, and `reg_lambda`. Best trial was trial 37.

**Optimal hyperparameters found:**

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| `learning_rate` | 0.0162 | Faster convergence than the 0.01 default, reduces tree count from ~2000 to 1194 |
| `max_depth` | 8 | Deeper trees capture more complex feature interactions |
| `subsample` | 0.650 | Trains each tree on 65% of rows — reduces overfitting |
| `colsample_bytree` | 0.839 | Each tree uses ~84% of features |
| `min_child_weight` | 3 | Minimum samples per leaf — regularises against overfitting |
| `reg_alpha` | 0.483 | L1 regularisation — prunes weak feature contributions |
| `reg_lambda` | 2.515 | L2 regularisation — smooths leaf weights |

The higher `max_depth=8` combined with strong regularisation (`reg_alpha`, `min_child_weight`) is a consistent pattern — deeper trees with regularisation outperform shallow unregularised trees on this dataset. The faster learning rate means the model converges in 1,194 trees rather than nearly 2,000, making it ~40% faster at inference while achieving better generalisation.

---

### Feature Importances (V8)

The top features by mean importance across 5 CV folds:

1. `lag1` — same hour last year, the strongest single signal
2. `cooling_degrees` — heat stress above 18.3°C
3. `lag3` — same hour three years ago
4. `lag2` — same hour two years ago
5. `rolling_mean_24h` — yesterday's average demand
6. `temperature` — raw regional temperature
7. `hour` — time of day

Calendar features (`month`, `weekofyear`, `dayofweek`, `dayofyear`, `year`) have near-zero importance individually but contribute to the overall model through feature interactions at deeper tree levels.

---

### Error Analysis

The model performs worst on days involving sudden weather-driven demand surges — specifically:
- Sharp cold snaps where temperatures drop significantly within 24 hours
- Unexpected heatwaves at the start of summer

These events are difficult to predict because `lag1` looks up the same period from last year (which may not have experienced equivalent weather), and `rolling_mean_24h` has not yet registered the spike at the start of the event. This is the **fundamental limitation** of a purely lag and calendar-based model without forecast temperature data — it can react to demand shifts via rolling features but cannot fully anticipate them.

---

### Limitations

- **Reactionary rather than anticipatory**: The model uses historical temperature, not forecast temperature. In production, feeding in a 24-hour weather forecast as input would significantly improve spike prediction.
- **No exogenous demand drivers**: Industrial output, gas prices, and grid topology changes are not captured.
- **Training data ceiling**: The processed dataset covers 2004–2018. The model may not generalise well to post-2018 consumption patterns, which have been further shaped by remote work, EV adoption, and efficiency improvements.

---

### Potential Improvements

- **Forecast temperature as input** — replace historical temperature with a 24-hour weather forecast at inference time to anticipate rather than react to demand spikes
- **SHAP analysis** — SHapley Additive Explanations provide richer interpretability than XGBoost's built-in importance scores, quantifying each feature's directional contribution per prediction
- **LightGBM benchmark** — comparable accuracy to XGBoost with significantly faster training
- **Additional exogenous variables** — regional natural gas prices, industrial production indices

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost optuna requests
```
