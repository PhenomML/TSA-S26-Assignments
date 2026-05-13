# Homework 3 — Frequency Domain Analysis and State-Space Models

**Stat 207, Spring 2026**
**Due: May 21, 2026**

---

## Overview

This assignment develops two major frameworks from Module 2:

- **Problems 1–2** — Frequency domain: periodogram, spectral smoothing, AR spectral estimation, and coherence.
- **Problem 3** — The EWMA–ARIMA–State-Space connection: a four-part derivation that unifies ideas across Modules 1 and 2.
- **Problems 4–5** — Kalman filtering and smoothing applied to a real economic series.

## Series Assignment

Each student analyzes a unique US state monthly unemployment rate from FRED, plus the national series for comparison. Locate your series by alphabetical rank of your last name.

| Rank | State | FRED ID |
|------|-------|---------|
| 1–4 | California | `CAUR` |
| 5–8 | Texas | `TXUR` |
| 9–12 | New York | `NYUR` |
| 13–16 | Michigan | `MIUR` |
| 17–20 | Colorado | `COUR` |
| 21–25 | Washington | `WAUR` |

## Setup

```r
if (!requireNamespace("astsa", quietly = TRUE))
  install.packages("astsa", repos = "https://cran.r-project.org")

# Download data from FRED (no API key required)
read_fred <- function(id, start = "1990-01-01", end = "2024-12-01") {
  url <- paste0("https://fred.stlouisfed.org/graph/fredgraph.csv?id=", id)
  df  <- read.csv(url, colClasses = c("Date", "numeric"))
  df  <- df[df[[1]] >= as.Date(start) & df[[1]] <= as.Date(end), ]
  ts(df[[2]], start = c(1990, 1), frequency = 12)
}

library(astsa)

# Replace CAUR with your assigned ID
x <- read_fred("CAUR")   # state unemployment rate, monthly, 1990-2024
u <- read_fred("UNRATE") # national unemployment rate, for Problem 2
```

---

## Problem 1 — Spectral Analysis (35 pts)

**(a)** [5 pts] Plot `x`. Comment on any visible trend, seasonal pattern, or business-cycle variation. Also plot the sample ACF out to lag 48 and describe the dominant features.

**(b)** [12 pts] Compute and plot the raw periodogram using `mvspec(x, log = "no")`.

- Identify the **three largest** periodogram ordinates. Express each as (i) cycles per month and (ii) cycle length in months.
- The annual seasonal cycle corresponds to $\omega = 1/12 \approx 0.083$ cycles/month. Is it visible? If absent or weak, explain why.

**(c)** [12 pts] Compute smoothed periodograms with $L = 5$, $L = 15$, and $L = 51$ using `mvspec(x, spans = L, log = "yes", taper = 0)`. Superimpose all three on a single log-scale figure with 95% confidence intervals.

- The smoothed periodogram follows an approximate $\chi^2_{2L}$ distribution. Use this to explain why larger $L$ produces narrower confidence intervals.
- Choose the $L$ you consider best-suited for this series and justify your choice in terms of the bias–variance tradeoff.

**(d)** [6 pts] Compute the AR spectral estimate via `spec.ar(x)`. Which model order does AIC select? Identify the two dominant peaks and compare their locations and heights to the smoothed estimate from part (c).

---

## Problem 2 — Coherence (20 pts)

**(a)** [5 pts] Standardize both `x` (state) and `u` (national) to zero mean and unit variance. Plot them on the same axes. During which time periods do they diverge most visibly?

**(b)** [10 pts] Compute and plot the squared coherence using `mvspec(cbind(x, u), spans = 15)`. The squared coherence is stored in `$coh` on the returned object.

- At which frequencies is coherence highest? Express as cycle lengths in months.
- At which frequencies is coherence low? What does low coherence indicate about the state-vs-national relationship at those frequencies?

**(c)** [5 pts] High squared coherence at frequency $\omega$ means both series share a strong common oscillation at period $1/\omega$. Does this imply that national unemployment *causes* state unemployment? Explain briefly (≤ 100 words).

---

## Problem 3 — The EWMA–ARIMA–State-Space Trinity (25 pts)

This problem traces a single idea through three equivalent representations. No R is required; show all algebra.

The **exponentially weighted moving average** (EWMA) level at time $t$ is defined by:

$$S_t = \lambda\, y_t + (1 - \lambda)\, S_{t-1}, \quad \lambda \in (0, 1)$$

**(a)** [6 pts] **EWMA as ARIMA.** Consider the IMA(1,1) model

$$\nabla y_t = w_t + \theta\, w_{t-1}, \quad w_t \sim \text{iid } N(0, \sigma^2_w)$$

The optimal one-step-ahead MMSE forecast satisfies $\hat{y}_{t+1|t} = \hat{y}_{t|t-1} + (1+\theta)(y_t - \hat{y}_{t|t-1})$.

Show that $\hat{y}_{t+1|t} = S_t$ (the EWMA level) and identify the relationship between $\lambda$ and $\theta$. For which values of $\theta$ is the implied $\lambda$ in $(0, 1)$?

**(b)** [6 pts] **EWMA as state-space.** Write the **local level model**

$$y_t = \mu_t + v_t, \qquad \mu_t = \mu_{t-1} + w_t$$

with $v_t \sim N(0, \sigma^2_v)$ and $w_t \sim N(0, \sigma^2_w)$ independent, in standard state-space form $\mathbf{x}_t = \Phi\, \mathbf{x}_{t-1} + \mathbf{w}_t$, $y_t = A\, \mathbf{x}_t + v_t$. Identify $\Phi$, $A$, $Q = \text{Cov}(\mathbf{w}_t)$, and $R = \text{Cov}(v_t)$.

**(c)** [7 pts] **Kalman filter = EWMA.** The Kalman filter for the local level model is:

$$\mu_t^{t-1} = \mu_{t-1}^{t-1}, \qquad P_t^{t-1} = P_{t-1}^{t-1} + \sigma^2_w$$

$$K_t = \frac{P_t^{t-1}}{P_t^{t-1} + \sigma^2_v}, \qquad \mu_t^t = \mu_t^{t-1} + K_t\bigl(y_t - \mu_t^{t-1}\bigr), \qquad P_t^t = (1 - K_t)\,P_t^{t-1}$$

- Show that when $K_t$ converges to a steady-state value $K^*$, the update equation becomes exactly the EWMA recursion with $\lambda = K^*$.
- By imposing stationarity ($P_t^t = P_{t-1}^{t-1}$) on the Kalman recursions, derive the equation that $K^*$ must satisfy. Write it as a polynomial in $K^*$.

**(d)** [6 pts] **When does the smoother beat the filter?** For the local level model in steady state, the smoothed state variance satisfies:

$$P^n = \frac{P_f \cdot P_p}{P_f + P_p}$$

where $P_f = (1-K^*)\,P_p$ is the (steady-state) filtered variance and $P_p = P_f + \sigma^2_w$ is the predicted variance.

- Verify that $P^n \leq P_f$ always.
- Compute $\lim_{\sigma^2_w/\sigma^2_v \to 0} P^n / P_f$. What does this say about the smoother's gain when $\lambda$ is very small?
- In one or two sentences: under what real-world conditions is the Kalman smoother significantly more accurate than the Kalman filter (EWMA), and why?

---

## Problem 4 — Kalman Filter Applied (10 pts)

Fit the local level model to `x` and run the Kalman filter.

**(a)** [3 pts] Estimate $\sigma^2_w$ and $\sigma^2_v$ by MLE:

```r
fit <- StructTS(x, type = "level")
sw  <- sqrt(fit$coef["level"])    # sigma_w
sv  <- sqrt(fit$coef["epsilon"])  # sigma_v
```

Report both estimates. Using the relationship $\lambda = K^*$ from Problem 3(c), what EWMA smoothing weight does the MLE imply? Does this suggest a smooth or volatile latent level?

**(b)** [7 pts] Run the Kalman filter and plot filtered state estimates $\mu_t^t$ with $\pm 2\sqrt{P_t^t}$ bands over the original series (in gray):

```r
kf <- Kfilter(x,
              A      = matrix(1, 1, 1),
              mu0    = matrix(x[1], 1, 1),
              Sigma0 = matrix(var(x), 1, 1),
              Phi    = matrix(1, 1, 1),
              sQ     = matrix(sw, 1, 1),
              sR     = matrix(sv, 1, 1))
# kf$Xf = filtered states;  sqrt(kf$Pf[1,1,]) = filtered SE
```

Comment on: (i) how the filter tracks the 2008–2009 recession spike, and (ii) how it behaves at the very start of the series before $K_t$ has converged to $K^*$.

---

## Problem 5 — Kalman Smoother (10 pts)

**(a)** [5 pts] Run the Kalman smoother with the same parameters:

```r
ks <- Ksmooth(x,
              A      = matrix(1, 1, 1),
              mu0    = matrix(x[1], 1, 1),
              Sigma0 = matrix(var(x), 1, 1),
              Phi    = matrix(1, 1, 1),
              sQ     = matrix(sw, 1, 1),
              sR     = matrix(sv, 1, 1))
# ks$Xs = smoothed states;  sqrt(ks$Ps[1,1,]) = smoothed SE
```

Plot filtered and smoothed estimates together (different colors). Where do they differ most?

**(b)** [5 pts] Based on the $\lambda$ you computed in Problem 4(a) and the theoretical result from Problem 3(d), was a large or small smoother improvement expected? Does the plot confirm this? Explain in ≤ 100 words.

---

## Submission

Submit a single knitted PDF from your `.Rmd` file (RStudio) or an executed `.ipynb` exported to PDF (Jupyter). All plots must have axis labels, titles, and a legend where multiple lines appear.
