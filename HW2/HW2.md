# Homework 2 — ARMA Identification in Practice

**Stat 207, Spring 2026**  
**Due: April 28, 2026**

---

## Overview

This assignment develops three complementary skills, each directly relevant to the midterm:

- **Problem 1** — Reading ACF and PACF plots to identify AR, MA, and ARMA structure in six real time series, then fitting models via AIC-guided order selection.
- **Problem 2** — Building intuition for the AR(2) stationarity region by simulating processes across a grid of parameter values, then applying the analytical conditions and connecting them to root structure.
- **Problem 3** — Detecting ARCH effects in residuals: the workflow for recognizing when a constant-variance model is inadequate.

---

## Setup

### RStudio users

Paste into the RStudio Console (not a script chunk) once:

```r
install.packages("remotes", repos = "https://cloud.r-project.org")
remotes::install_github("FinYang/tsdl")
install.packages("astsa")
```

### Jupyter users

This notebook runs in JupyterLab using the `stats207` conda environment,
which provides R, Jupyter, and the `ir` kernel together.

**Step 1 — Create and activate the environment** (once only, from the repo root):

```bash
conda env create -f environment.yml
conda activate stats207
```

**Step 2 — Register the R kernel with Jupyter** (once only):

```bash
Rscript -e "IRkernel::installspec(user=TRUE)"
```

**Step 3 — Install TSDL and astsa** (once only):

```bash
Rscript -e "remotes::install_github('FinYang/tsdl')"
Rscript -e "install.packages('astsa', repos='https://cloud.r-project.org')"
```

**Step 4 — Launch JupyterLab** (each session, with `stats207` active):

```bash
jupyter lab
```

### All users — load the libraries

Run this cell before any other code:

```r
library(tsdl)
library(astsa)
```

---

## Problem 1 — ACF/PACF Analysis of 6 Time Series

### Motivation

The midterm asks you to identify AR, MA, and ARMA structure from ACF and PACF plots.
These six series span the key patterns: stationary AR, near-white-noise MA, nonstationary
economic series, financial returns with potential ARCH effects, and a seasonal structure.

### 1a. Extract the Six Series

```r
selected_idx <- c(
  531,  # Annual snowfall, Buffalo, NY, 1910–1972    [likely stationary AR or ARMA]
  347,  # Annual rainfall, London, 1813–1912         [likely stationary, near WN or MA]
  570,  # US Gross National Product, 1874–1970       [nonstationary — needs log-diff]
  279,  # Annual changes in global temperature       [stationary, AR structure]
  516,  # Annual bond yield, 1900–1970               [nonstationary — needs diff]
  437   # Oil prices (constant 1997 $), 1870–1997    [nonstationary — needs log or diff]
)

tsdl_desc <- function(i) attr(tsdl[[i]], "description")
series <- lapply(selected_idx, function(i) tsdl[[i]])
labs   <- sapply(seq_along(selected_idx), function(k)
            paste0(k, ". ", substr(tsdl_desc(selected_idx[k]), 1, 30)))
```

### 1b. Plot All Six Series

```r
par(mfrow = c(2, 3), mar = c(3, 3, 3, 1))
for (k in seq_along(series))
  plot(series[[k]], main = labs[k], ylab = "", xlab = "")
```

**Question 1.1**

(a) Which of the six series appear nonstationary (trending mean or growing variance)?

(b) For each nonstationary series, propose a transformation — `diff()` or `diff(log())` —
    to achieve approximate stationarity. Justify your choice: when should you log-transform
    before differencing?

### 1c. Preprocessing

```r
# Modify transforms as needed based on your answer to Q1.1
transforms <- list(
  identity,                    # series 1 — Buffalo snowfall
  identity,                    # series 2 — London rainfall
  function(x) diff(log(x)),   # series 3 — GNP
  identity,                    # series 4 — temp changes
  diff,                        # series 5 — bond yield
  function(x) diff(log(x))    # series 6 — oil prices
)

series_proc <- mapply(function(x, f) f(x), series, transforms, SIMPLIFY = FALSE)
```

### 1d. ACF and PACF

```r
par(mfrow = c(4, 3), mar = c(2, 2, 2.5, 1), oma = c(0, 0, 4, 0))
for (k in 1:6)
  acf( series_proc[[k]], lag.max = 20, main = labs[k], cex.main = 0.7, ylab = "ACF")
for (k in 1:6)
  pacf(series_proc[[k]], lag.max = 20, main = "",        cex.main = 0.7, ylab = "PACF")
mtext("Figure 1 — ACF (top) and PACF (bottom): All Six Series",
      outer = TRUE, cex = 1.1, font = 2, line = 1)
```

**Question 1.2**

Using the identification rules from Lecture 5:

| | ACF | PACF |
|---|---|---|
| AR($p$) | Tails off | Cuts off after lag $p$ |
| MA($q$) | Cuts off after lag $q$ | Tails off |
| ARMA($p$,$q$) | Tails off | Tails off |

Fill in the table for each preprocessed series:

| Series | Transformation applied | ACF pattern | PACF pattern | Suggested ARMA($p$,$q$) |
|---|---|---|---|---|
| 1 — Buffalo snow | | | | |
| 2 — London rain | | | | |
| 3 — GNP growth | | | | |
| 4 — Temp changes | | | | |
| 5 — Bond yield $\nabla$ | | | | |
| 6 — Oil $\nabla\log$ | | | | |

### 1e. AIC-Guided Model Fitting

```r
fit_arma_aic <- function(x, max_p = 4, max_q = 4) {
  best_aic <- Inf; best_fit <- NULL; best_pq <- c(NA, NA)
  for (p in 0:max_p) for (q in 0:max_q) {
    if (p == 0 && q == 0) next
    fit <- tryCatch(arima(x, order = c(p, 0, q), method = "ML"),
                   warning = function(w) NULL, error = function(e) NULL)
    if (!is.null(fit) && AIC(fit) < best_aic) {
      best_aic <- AIC(fit); best_fit <- fit; best_pq <- c(p, q)
    }
  }
  list(p = best_pq[1], q = best_pq[2], AIC = best_aic, fit = best_fit)
}

fits <- lapply(series_proc, fit_arma_aic)

cat(sprintf("%-4s %-5s %-5s %-10s\n", "Ser", "p", "q", "AIC"))
for (k in seq_along(fits)) {
  r <- fits[[k]]
  cat(sprintf("%-4d %-5s %-5s %-10s\n", k,
    ifelse(is.na(r$p), "—", r$p), ifelse(is.na(r$q), "—", r$q),
    ifelse(is.infinite(r$AIC), "FAILED", round(r$AIC, 2))))
}
```

**Question 1.3**

(a) Does the AIC-selected $(p, q)$ agree with your visual identification from Q1.2?
    Discuss any disagreement for at least one series.

(b) For the AR($p$) fits ($q = 0$), verify causality: compute the roots of
    $\Phi(z) = 1 - \phi_1 z - \cdots - \phi_p z^p$ and confirm all lie outside the unit circle.

```r
check_causality <- function(fit, label = "") {
  ar_coefs <- fit$coef[grep("^ar", names(fit$coef))]
  if (length(ar_coefs) == 0) { cat(label, ": pure MA\n"); return(invisible()) }
  roots <- polyroot(c(1, -ar_coefs))
  cat(label, ": root moduli =", round(Mod(roots), 3),
      "  causal:", all(Mod(roots) > 1), "\n")
}
for (k in seq_along(fits))
  if (!is.null(fits[[k]]$fit)) check_causality(fits[[k]]$fit, paste("Series", k))
```

---

## Problem 2 — AR(2) Parameter Space

Consider 25 AR(2) processes, each satisfying:

$$y_{t+1} = \phi_1 y_t + \phi_2 y_{t-1} + w_t, \qquad w_t \sim \text{WN}(0,1)$$

where $\phi_1$ and $\phi_2$ each take five values:
$\{-1.25,\; -0.625,\; 0,\; 0.625,\; 1.25\}$,
forming a $5 \times 5$ grid of 25 parameter combinations.

### 2a. Simulate and Plot

```r
phi_vals <- seq(-1.25, 1.25, length.out = 5)

sim_ar2 <- function(phi1, phi2, n = 200, seed = 42) {
  set.seed(seed)
  x <- numeric(n + 2)
  for (t in 3:(n + 2)) {
    x[t] <- phi1 * x[t-1] + phi2 * x[t-2] + rnorm(1)
    if (!is.finite(x[t])) {
      x[t:(n+2)] <- NA
      break
    }
  }
  ts(x[3:(n + 2)])
}

par(mfrow = c(5, 5), mar = c(1, 1, 2.2, 0.5), oma = c(3, 3, 5, 1))

for (phi2 in rev(phi_vals)) {
  for (phi1 in phi_vals) {
    plot(sim_ar2(phi1, phi2), type = "l", col = "black", lwd = 0.8,
         axes = FALSE, xlab = "", ylab = "",
         main = bquote(phi[1]==.(phi1)~","~phi[2]==.(phi2)),
         cex.main = 0.58)
    box()
  }
}

mtext(expression(phi[1]), side = 1, outer = TRUE, line = 1.5, cex = 1.1)
mtext(expression(phi[2]), side = 2, outer = TRUE, line = 1.5, cex = 1.1)
mtext("Figure 2 — AR(2) Parameter Space",
      outer = TRUE, cex = 1.05, font = 2, line = 2)
```

**Question 2.1**
From Figure 2, list the $(\phi_1, \phi_2)$ combinations whose simulated
paths *appear* stationary (fluctuating around a fixed mean with bounded
variance throughout the 200-observation run).

### 2b. Analytical Stationarity

The three conditions for AR(2) stationarity (S&S eq. 3.28, Lecture 5) are:

$$\phi_1 + \phi_2 < 1 \qquad \phi_2 - \phi_1 < 1 \qquad |\phi_2| < 1$$

```r
is_stationary_ar2 <- function(phi1, phi2) {
  (phi1 + phi2 < 1) && (phi2 - phi1 < 1) && (abs(phi2) < 1)
}

cat("Analytical stationarity (TRUE = all three conditions met):\n\n")
cat(sprintf("%12s", "phi2 \\ phi1"))
for (p1 in phi_vals) cat(sprintf("%9.4g", p1))
cat("\n", strrep("-", 12 + 9 * 5), "\n")
for (phi2 in rev(phi_vals)) {
  cat(sprintf("%12.4g", phi2))
  for (phi1 in phi_vals)
    cat(sprintf("%9s", if (is_stationary_ar2(phi1, phi2)) "TRUE" else "FALSE"))
  cat("\n")
}
```

**Question 2.2**

(a) How many of the 25 combinations are analytically stationary?
    Verify each of the three conditions individually for all 25 cases
    and summarise your results in a table.

(b) Do the analytical results match your visual identification from
    Question 2.1? Comment on any near-boundary cases — do processes
    near the stationarity boundary look obviously stationary in a
    200-observation run?

**Question 2.3**
Among the stationary processes, some look smooth and others oscillate.

(a) The character of an AR(2) is determined by whether the roots of
    $\Phi(z) = 1 - \phi_1 z - \phi_2 z^2$ are real or complex.
    For the oscillatory stationary cases, compute the roots and the
    implied cycle period $T = 2\pi / \text{Arg}(z_0)$:

```r
ar2_roots <- function(phi1, phi2) {
  roots        <- polyroot(c(1, -phi1, -phi2))
  moduli       <- Mod(roots)
  complex_root <- roots[Im(roots) > 1e-10]
  period       <- if (length(complex_root) > 0) 2 * pi / Arg(complex_root[1]) else NA
  list(roots = roots, moduli = moduli, period = period)
}

# Replace this list with the stationary combinations you identified in Q2.1/2.2
my_stationary <- list(
  c(phi1 = 0, phi2 = 0)   # example — add your combinations here
)

cat(sprintf("%-8s %-8s %-14s %-14s %-8s\n",
            "phi1", "phi2", "root moduli", "complex?", "period"))
cat(strrep("-", 56), "\n")
for (pq in my_stationary) {
  r <- ar2_roots(pq["phi1"], pq["phi2"])
  cat(sprintf("%-8.4g %-8.4g %-14s %-14s %-8s\n",
              pq["phi1"], pq["phi2"],
              paste(round(r$moduli, 3), collapse = ", "),
              any(abs(Im(r$roots)) > 1e-10),
              ifelse(is.na(r$period), "—", round(r$period, 2))))
}
```

(b) Does the computed period match what you observe visually in Figure 2?
    Estimate the period by counting peaks in the relevant panels and compare.

**Question 2.4 — Synthesis**

Using the material from Lecture 5, explain in 2–3 sentences why the
stationarity region in the $(\phi_1, \phi_2)$ plane has the triangular
shape seen in Figure 2. Why are rows with $|\phi_2| = 1.25$ always
non-stationary regardless of $\phi_1$?

**Question 2.5 — Analytical Practice**

For each of the following $(\phi_1, \phi_2)$ pairs, apply the three
stationarity conditions analytically (do not simulate) and state whether
the process is stationary. If not, identify which condition(s) fail.

| $\phi_1$ | $\phi_2$ | Stationary? | Which condition(s) fail, if any? |
|---|---|---|---|
| 0.5 | 0.4 | | |
| 1.0 | −0.3 | | |
| −0.8 | −0.3 | | |
| 0.6 | 0.5 | | |

---

## Problem 3 — Detecting ARCH Structure

### 3a. US GNP Growth Rate

The `gnp` dataset in the `astsa` package contains US quarterly GNP from 1947 to 2002.

```r
gnpgr <- diff(log(gnp))
plot(gnpgr, main = "US GNP Log-Growth Rate", ylab = "log-growth rate")
```

**Question 3.1** Does `gnpgr` appear stationary? Describe the mean and variance behavior visually.

### 3b. Fit an AR(1) Mean Model

```r
fit_ar1 <- arima(gnpgr, order = c(1, 0, 0))
cat("AR(1) coefficients:\n"); print(coef(fit_ar1))

resid_ar1 <- residuals(fit_ar1)
```

**Question 3.2** Report the estimated AR(1) coefficient $\hat{\phi}$ and its standard error.
Is it significantly different from zero?

### 3c. Check for ARCH Effects

ARCH effects show up as autocorrelation in the *squared* residuals.

```r
par(mfrow = c(1, 2))
acf( resid_ar1^2, lag.max = 12, main = "ACF of Squared Residuals")
pacf(resid_ar1^2, lag.max = 12, main = "PACF of Squared Residuals")
```

**Question 3.3**

(a) Do the squared residuals show autocorrelation? At which lags?

(b) What does this indicate about the adequacy of the constant-variance AR(1) model?

(c) A classmate says: "The Ljung-Box test on the AR(1) residuals already passed —
    why do we need to check the squared residuals too?" Explain the distinction.

### 3d. Interpretation

**Question 3.4**

The ARCH(1) model adds $\sigma^2_t = \alpha_0 + \alpha_1 \hat{e}^2_{t-1}$ to account
for time-varying variance. Suppose you fit this model and obtain
$\hat{\alpha}_1 = 0.19$ and $\hat{\alpha}_0 = 7.3 \times 10^{-5}$.

(a) Compute the unconditional variance $E(y^2_t)$. Is the model stationary?

(b) If a quarter had an unusually large $|\text{gnpgr}_t|$ (e.g., during a recession),
    what does the ARCH(1) model predict about GNP growth variance in the *following* quarter?

```r
# Optional: fit ARCH(1) using garchFit (fGarch package)
# library(fGarch)
# summary(garchFit(~arma(1,0)+garch(1,0), gnpgr))
```
