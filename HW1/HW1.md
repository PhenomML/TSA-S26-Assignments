# Homework 1: Foundations of Time Series Analysis

**Stanford Time Series Analysis — Spring 2026**
**Due: April 21, 2026, 11:59 PM**

Submit your completed R Markdown (knitted to PDF or HTML) or Jupyter notebook (with R kernel) to Canvas. All code must be reproducible. Begin by loading the `astsa` package.

```r
library(astsa)
```

---

## Problem 1: Exploring Time Series Data (20 pts)
*S&S §1.2–1.5*

### (a) [4 pts]
Plot the Johnson & Johnson quarterly earnings per share series (`jj`) and the Southern Oscillation Index (`soi`) using `tsplot()`. For each series, describe the following visual features:
- Is there an apparent trend?
- Is there apparent seasonality or periodicity?
- Does the variance appear constant over time?

### (b) [4 pts]
Compute and plot the sample ACF for both `jj` and `soi` using `acf1()` (or `acf2()`). Based on the ACF alone, does either series appear to be white noise? Justify your answer by referencing the approximate 95% confidence bounds.

### (c) [6 pts]
For the `jj` series:

(i) Plot both `jj` and `log(jj)` side-by-side. How does the log transformation affect the variance of the series?

(ii) Compute the sample ACF of `log(jj)`. Comment on what the pattern reveals about the correlation structure (hint: consider lags that are multiples of 4).

### (d) [6 pts]
Compute the sample cross-correlation function (CCF) between `soi` and `rec` (Recruitment) using `ccf()` or `acf2()`.

(i) At what lag does the cross-correlation appear strongest? Is `soi` leading or lagging `rec`?

(ii) Based on the CCF, what is a plausible physical interpretation of the relationship between these two series?

---

## Problem 2: Detrending and Transformations (25 pts)
*S&S §2.2; Lecture 3*

Use the global land temperature anomaly series `gtemp_land` throughout this problem.

### (a) [5 pts]
Plot `gtemp_land`. Fit a simple linear trend model
$$x_t = \beta_0 + \beta_1\, t + w_t$$
using `lm()` or `trend()` from `astsa`. Superimpose the fitted trend line on the data. What is the estimated annual rate of warming (in degrees Celsius per year)? Is it statistically significant?

### (b) [5 pts]
Plot the residuals from the fit in (a). Compute their sample ACF. Do the residuals appear to be white noise? What does this imply about the adequacy of the linear trend model?

### (c) [5 pts]
Compute the first-differenced series $\nabla x_t = x_t - x_{t-1}$ using `diff()`. Plot it and compute its sample ACF. How does differencing compare to regression detrending as a means of producing a stationary series?

### (d) [5 pts]
Fit three nested regression models to `gtemp_land`:

| Model | Formula |
|-------|---------|
| 1 | $x_t = \beta_0 + \beta_1 t + w_t$ |
| 2 | $x_t = \beta_0 + \beta_1 t + \beta_2 t^2 + w_t$ |
| 3 | $x_t = \beta_0 + \beta_1 t + \beta_2 t^2 + \beta_3 t^3 + w_t$ |

For each model, compute the AIC, AICc, and BIC using the formulas from Lecture 3:
$$\text{AIC}(k) = \log\hat{\sigma}_k^2 + \frac{n+2k}{n}, \quad \text{BIC}(k) = \log\hat{\sigma}_k^2 + \frac{k\log n}{n}$$
Which model does each criterion select? Do they agree?

### (e) [5 pts]
Apply two smoothing methods to `gtemp_land`:

(i) A symmetric moving average smoother using `filter()` with a span of your choice.

(ii) A lowess smoother using `trend(gtemp_land, lowess=TRUE)`.

Plot both smooths superimposed on the original data. Which smoother do you prefer for communicating the long-term warming trend, and why?

---

## Problem 3: AR, MA, and ARMA Models (30 pts)
*S&S §3.1–3.3*

### (a) [8 pts]
Simulate four AR(1) series of length $n = 200$ using `sarima.sim()` with $\phi \in \{0.9,\ 0.5,\ -0.5,\ -0.9\}$ and $\sigma_w^2 = 1$.

For each series:
- Plot the simulated path.
- Compute the sample ACF and overlay the theoretical ACF $\rho(h) = \phi^h$ for $h = 0, 1, \ldots, 20$.

Comment on how the sign and magnitude of $\phi$ affect (i) the visual appearance of the series and (ii) the rate of ACF decay.

### (b) [6 pts]
Simulate an MA(1) series of length $n = 200$ with $\theta = 0.8$ and $\sigma_w^2 = 1$. Compute the sample ACF and PACF using `acf2()`.

(i) At what lag does the theoretical ACF cut off? What is the theoretical value of $\rho(1)$?

(ii) Contrast the ACF and PACF patterns of an MA(1) with those of an AR(1) (from part (a)). How does Table 3.1 (S&S p. 111) help distinguish the two model families?

### (c) [8 pts]
Consider the AR(2) process
$$x_t = 1.5\, x_{t-1} - 0.75\, x_{t-2} + w_t, \quad \sigma_w^2 = 1.$$

(i) Show analytically that this model is causal by finding the roots of $\phi(z) = 1 - 1.5z + 0.75z^2$ and verifying that they lie outside the unit circle. What is the implied pseudo-period of the oscillation?

(ii) Simulate $n = 144$ observations and plot the series. Describe its visual character.

(iii) Compute the sample ACF and PACF using `acf2()`. Does the PACF cut off after lag 2 as expected for an AR(2)?

*Hint:* See S&S Example 3.12 and Fig. 3.5 for the expected patterns.

### (d) [8 pts]
Use the Recruitment series `rec`.

(i) Plot the sample ACF and PACF using `acf2(rec, 48)`.

(ii) Based on the ACF/PACF patterns and Table 3.1, propose a model class and order for `rec`. Justify your choice.

(iii) Fit your proposed model using `ar.yw()` (Yule-Walker estimation). Report the estimated coefficients and $\hat{\sigma}_w^2$. Compare with the results in S&S Example 3.27.

---

## Problem 4: Forecasting (25 pts)
*S&S §3.4*

Use the Recruitment series `rec` throughout.

### (a) [6 pts]
Fit an AR(2) model to `rec` using `ar.ols()` with `demean=FALSE, intercept=TRUE` (as in S&S Example 3.17). Report the fitted model equation:
$$x_{t+1}^n = \hat\phi_0 + \hat\phi_1 x_t + \hat\phi_2 x_{t-1}.$$
Compare your OLS estimates with the Yule-Walker estimates from Problem 3(d). Are they similar?

### (b) [8 pts]
Produce 24-month-ahead forecasts from your fitted AR(2) model using `predict()` (or `sarima.for()`). Plot the forecasts with ±1 SE prediction bands superimposed on the last few years of data (as in S&S Fig. 3.7).

(i) What happens to the point forecast as the horizon $m$ grows? Explain in terms of the theoretical result $x_{n+m}^n \to \mu_x$ as $m \to \infty$ (S&S eq. 3.83).

(ii) What happens to the width of the prediction intervals as $m$ grows?

### (c) [7 pts]
The mean-square $m$-step prediction error for a causal ARMA model is (S&S eq. 3.81):
$$P_{n+m}^n = \sigma_w^2 \sum_{j=0}^{m-1} \psi_j^2$$

(i) Compute the first 24 $\psi$-weights for your fitted AR(2) using `ARMAtoMA(ar=c(phi1, phi2), ma=0, 24)`. Plot them.

(ii) Using the $\psi$-weights, manually compute $P_{n+m}^n$ for $m = 1, 2, \ldots, 24$ and plot $\sqrt{P_{n+m}^n}$ (the forecast standard error) versus $m$. Does this match the ±1 SE bands from part (b)?

### (d) [4 pts] Bonus
Compute forecasts for $m = 1, 2, \ldots, 100$ steps ahead. Plot the forecast values $x_{n+m}^n$ versus $m$. At approximately what horizon does the forecast come within 1% of $\bar{x}$ (the sample mean of `rec`)? How does this connect to the rate of $\psi$-weight decay?

---

*Total: 100 pts (+ 4 bonus)*
