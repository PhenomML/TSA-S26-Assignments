# Time Series Analysis — Spring 2026

**Stanford University**

## Assignments

| Assignment | Due Date | Topics |
|------------|----------|--------|
| [HW1](HW1/) | April 21, 2026 | Stationarity, ACF, detrending, AR/MA/ARMA, forecasting (S&S Ch. 1-3) |
| [HW2](HW2/) | May 7, 2026 | ARMA identification, AR(2) stationarity, ARCH/GARCH detection (S&S Ch. 3, 5.4) |
| [HW3](HW3/) | May 21, 2026 | Spectral analysis, coherence, EWMA–ARIMA–state-space trinity, Kalman filter and smoother (S&S Ch. 4, 6) |

## Setup

Assignments use **R** with the [`astsa`](https://github.com/nickpoison/astsa) and [`tsdl`](https://github.com/FinYang/tsdl) packages. Install them once:

```r
install.packages("astsa")
install.packages("remotes")
remotes::install_github("FinYang/tsdl")
```

## Submission

Submit your completed Jupyter notebook (`.ipynb`, R kernel, all cells run with output visible) or knitted R Markdown to **Canvas** by the due date.

## Resources

- Shumway & Stoffer, *Time Series Analysis and Its Applications* (2025)
- [`astsa` package documentation](https://github.com/nickpoison/astsa)
