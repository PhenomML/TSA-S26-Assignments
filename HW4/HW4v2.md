# Homework 4 — Classical and Foundation Model Forecasting

**Stat 207, Spring 2026**
**Due: June 2, 2026**

---

## Overview

This assignment compares three forecasting paradigms across a common set of economic time series:

- **ARIMA** — classical linear model, fitted to each target series.
- **Hamilton Markov Switching** — regime-aware classical model (Lecture 15); explicitly models recession/growth dynamics.
- **Chronos** — large pretrained foundation model; zero-shot forecasting without fitting to the target.

You will benchmark all three on your assigned state unemployment series, stress-test them on series with contrasting structural properties, and critically compare the three paradigms.

**Language:** Python 3.11+. This assignment uses Python throughout; no R is required.

---

## Environment Setup

**Recommended: Google Colab** (free GPU, pre-installed PyTorch).

1. Open [colab.research.google.com](https://colab.research.google.com) and create a new notebook.
2. In the first code cell, run:

```python
!pip install chronos-forecasting statsmodels pmdarima pandas matplotlib
```

**Local alternative** (CPU is sufficient for `chronos-t5-tiny`):

```bash
conda create -n ts_foundation python=3.11
conda activate ts_foundation
pip install chronos-forecasting statsmodels pmdarima pandas matplotlib
```

---

## Series Assignment

Continue using the same state unemployment series as HW3. Download directly from FRED (no API key required):

```python
import pandas as pd

def get_fred(series_id, start="1990-01-01", end="2024-12-01"):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df  = pd.read_csv(url)
    date_col = df.columns[0]          # 'observation_date' in current FRED format
    df[date_col] = pd.to_datetime(df[date_col])
    df  = df.set_index(date_col)
    return df.loc[start:end].squeeze().asfreq("MS")

# Replace with your assigned state series from HW3
x = get_fred("CAUR")   # e.g., California unemployment rate
```

**Train/test split used throughout:** train on 1990–2021 (384 months), hold out 2022–2024 (36 months) as the test set.

```python
x_train = x[:"2021-12-01"]
x_test  = x["2022-01-01":]
```

---

## Problem 1 — Three-Way Benchmark (40 pts)

**(a)** [8 pts] Fit a **seasonal ARIMA baseline** on the training set using `pmdarima.auto_arima`:

```python
import pmdarima as pm

arima = pm.auto_arima(x_train, seasonal=True, m=12,
                      stepwise=True, suppress_warnings=True,
                      information_criterion="aic")
fc_arima, ci_arima = arima.predict(n_periods=36, return_conf_int=True, alpha=0.05)
```

Report the selected $(p, d, q)(P, D, Q)_{12}$ orders and the AIC of the fitted model.

**(b)** [14 pts] Fit a **2-regime Markov Switching AR** model (Hamilton 1989) on the training set. Following Hamilton's original approach, model the **first difference** $\Delta u_t = u_t - u_{t-1}$; regimes are distinguished by their volatility (crisis vs. normal):

```python
import numpy as np
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

# Hamilton-style: model month-over-month change in unemployment
dx_train = x_train.diff().dropna()

ms_model = MarkovAutoregression(
    dx_train, k_regimes=2, order=1,
    switching_ar=False,      # shared AR coefficient
    switching_variance=True, # regimes differ in volatility
)
np.random.seed(42)
ms_res = ms_model.fit(em_iter=50, search_reps=20)
print(ms_res.summary())

# Regime characteristics
params = ms_res.params
mu0,  mu1  = params['const[0]'],           params['const[1]']
sig0, sig1 = np.sqrt(params['sigma2[0]']), np.sqrt(params['sigma2[1]'])
phi1       = params['ar.L1']
p00,  p10  = params['p[0->0]'],            params['p[1->0]']
print(f"Regime 0 (Δu): mean={mu0:.4f}  sigma={sig0:.4f}")
print(f"Regime 1 (Δu): mean={mu1:.4f}  sigma={sig1:.4f}")
print(f"Expected crisis-regime duration: {1/(1-p00):.1f} months")
print(f"Expected normal-regime duration: {1/p10:.1f} months")

# Filtered regime probabilities at end of training
regime_probs = ms_res.filtered_marginal_probabilities.iloc[-1]
print(f"Filtered regime probs at end of 2021: {regime_probs.values.round(3)}")

# statsmodels does not support out-of-sample predict() for MarkovAutoregression;
# use a manual multi-step recursion instead.
T_mat = np.array([[p00, 1-p00], [p10, 1-p10]])
pi_T  = regime_probs.values

history_dx = [dx_train.iloc[-1]]
pi_h = pi_T.copy()
fc_dx_vals = []
for h in range(36):
    pi_h        = pi_h @ T_mat                     # propagate regime probs one step
    expected_mu = pi_h[0]*mu0 + pi_h[1]*mu1        # weighted intercept
    ar_contrib  = phi1 * history_dx[-1]
    fc = expected_mu + ar_contrib
    fc_dx_vals.append(fc)
    history_dx.append(fc)

# Integrate differences back to level forecasts
fc_ms = pd.Series(
    x_train.iloc[-1] + np.cumsum(fc_dx_vals),
    index=x_test.index
)
```

- Identify which regime has higher volatility (larger $\sigma$). Hamilton's (1989) original result identified regimes by their mean growth rate; here volatility is the primary identifier. Interpret the two regimes in economic terms (crisis vs. normal).
- Report the estimated expected duration of each regime (in months). Does the crisis-regime duration match the typical length of a US recession?
- Plot the **filtered regime probabilities** over the full training period (1990–2021). Do the inferred high-volatility periods align with known crises (1991 recession, 2008–09 GFC, COVID-19 2020)?
- What is the estimated probability of each regime at the end of 2021?

**(c)** [8 pts] Generate 36-month-ahead **Chronos** zero-shot forecasts:

```python
import torch
from chronos import ChronosPipeline

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cpu",          # change to "cuda" on Colab
    dtype=torch.float32,
)

context  = torch.tensor(x_train.values, dtype=torch.float32).unsqueeze(0)
forecast = pipeline.predict(context, prediction_length=36, num_samples=200)

median = forecast[0].median(dim=0).values.numpy()
low    = forecast[0].quantile(0.025, dim=0).numpy()
high   = forecast[0].quantile(0.975, dim=0).numpy()
```

**(d)** [6 pts] Plot the training series (1990–2021), the held-out test values (2022–2024), and all three forecasts on a single figure. Show 95% intervals for ARIMA and Chronos; plot the Markov Switching forecast as a dashed line (point forecasts only).

Compute MAE and RMSE for all three models on the 36-month test period and report in a table:

$$\text{MAE} = \frac{1}{36}\sum_{h=1}^{36}|\hat{y}_{T+h} - y_{T+h}|, \qquad \text{RMSE} = \sqrt{\frac{1}{36}\sum_{h=1}^{36}(\hat{y}_{T+h} - y_{T+h})^2}$$

| Model | MAE | RMSE |
|-------|-----|------|
| Seasonal ARIMA | | |
| Markov Switching AR | | |
| Chronos | | |

**(e)** [4 pts] Interpret the results. Hamilton's model explicitly represents regime structure; Chronos implicitly learned from many time series including recessions. Does the evidence from your series suggest that Chronos has internalized regime-like behavior, or does the explicit Hamilton model have an advantage? Explain in ≤ 100 words.

---

## Problem 2 — Stress-Testing All Three Models (30 pts)

Apply all three models (ARIMA, Markov Switching AR, Chronos) to three additional FRED series with structurally different characteristics, using the same 2022–2024 holdout. Use the same code as Problem 1, replacing `x_train`/`x_test` with each new series.

| Series | FRED ID | Structural characteristic |
|--------|---------|--------------------------|
| US Housing Starts | `HOUST` | Strong seasonality; catastrophic regime break (2006–2009 crash) in training data |
| WTI Crude Oil Price | `MCOILWTICO` | High volatility, irregular price shocks; no dominant seasonality |
| 10-Year Treasury Yield | `GS10` | Near-unit-root financial rate; persistent regime changes in level |

**(a)** [12 pts] For each series, fit all three models and compute MAE and RMSE. Summarize in a table with rows = series, columns = (ARIMA MAE, MS MAE, Chronos MAE, ARIMA RMSE, MS RMSE, Chronos RMSE).

**(b)** [12 pts] Include one forecast plot per series showing all three models against the test actuals. For each plot, write 2–3 sentences: which model performs best and why might the structural characteristics favor or hurt each approach?

**(c)** [6 pts] Across Problems 1 and 2, identify the conditions under which Markov Switching outperforms both ARIMA and Chronos. When does it underperform? In what setting would you expect all three models to converge to similar accuracy?

---

## Problem 3 — Critical Comparison (30 pts)

**(a)** [12 pts] **Forced choice.** A regional hospital network wants to forecast monthly emergency department visits 6 months ahead, to plan nurse staffing. They have 4 years of monthly data (48 observations). No external predictors are available. Forecasts must be explainable to a non-technical hospital board.

Choose **one** of the three models — seasonal ARIMA, Markov Switching AR, or Chronos — for this application. Write a structured argument (≤ 300 words) covering:

- Forecast accuracy given the small sample size
- Whether regime-switching behavior is plausible in this domain
- Interpretability and explainability to non-statisticians
- Uncertainty quantification for operational planning
- Ease of updating the model as new monthly data arrives

**(b)** [10 pts] Foundation models are pretrained on large, diverse corpora of time series. This creates a **distribution shift** risk: if the target series comes from a very different distribution than the pretraining data, zero-shot performance can degrade. Markov Switching models face a different risk: they assume a fixed number of regimes and a stationary transition matrix, which may fail when a new, previously unseen regime emerges (e.g., a pandemic).

Design an experiment (≤ 200 words) to audit *both* Chronos and the Markov Switching AR for these respective failure modes before deploying either in a high-stakes application. Specify:

- The datasets you would use (real or synthetic) and why
- The metrics and evaluation protocol
- A concrete success criterion for each model

**(c)** [8 pts] **Paradigm scorecard.** Based on your empirical results from Problems 1 and 2, rank the three models on each of the following dimensions. Justify each ranking in one sentence drawn from your results.

| Dimension | Ranking (1st → 3rd) | Justification |
|-----------|---------------------|---------------|
| Point forecast accuracy (MAE/RMSE) | | |
| Uncertainty quantification | | |
| Interpretability to a domain expert | | |
| Robustness to structural breaks | | |
| Ease of deployment with new data | | |

---

## Submission

Submit a Jupyter notebook (`.ipynb`) with **all cells executed** and outputs visible. Export to PDF and submit both `.ipynb` and `.pdf`. Narrative responses should appear in Markdown cells, not code comments. All plots must have axis labels, titles, and a legend where multiple lines appear.
