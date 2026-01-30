#  004 Pairs Trading with Cointegration & Kalman Filters
### ITESO — Market Microstructure and Trading Systems  
**Autors:**  
- José Armando Melchor Soto  
- Rolando Fortanell Canedo  

------------------------------------------------------------------------

#  Installation & Requirements

Clone the repository:

``` bash
git clone https://github.com/user/repository.git
cd repository
```

Create virtual environment (optional):

``` bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate    # Windows
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

If missing:

``` bash
pip install numpy pandas scipy statsmodels yfinance seaborn matplotlib ta
```

------------------------------------------------------------------------

#  System Architecture

    ├── libraries.py
    ├── main.py
    ├── main_trials.py
    ├── backtesting.py
    ├── cointegration.py
    ├── data_processing.py
    ├── kalman.py
    ├── metrics.py
    ├── classes.py
    ├── prints.py
    ├── visualization.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

#  Project Overview

This project develops a complete statistical arbitrage system using
pairs trading, cointegration testing, Kalman filters, and Sequential
Decision Analysis (SDA). The strategy identifies long-run equilibrium
relationships, monitors deviations in the spread, and executes dynamic
market-neutral positions based on Z-score signals.

------------------------------------------------------------------------

#  Objectives

-   Identify cointegrated equity pairs using rigorous statistical tests\
-   Implement Kalman Filters as dynamic state-space models\
-   Build a dynamic hedging, market-neutral strategy\
-   Evaluate performance with real transaction costs and borrow fees\
-   Maintain modular, clean, fully documented Python code\
-   Provide full analysis including Required Charts and Performance
    Metrics

------------------------------------------------------------------------

#  Key Concepts

-   **Cointegration**\
-   **Mean Reversion**\
-   **Dynamic Hedging via Kalman Filters**\
-   **VECM Error Correction**\
-   **SDA Sequential Decision Modeling**\
-   **Market Neutrality**

------------------------------------------------------------------------

#  Technical Requirements

### Data

-   15 years of daily price data\
-   60/20/20 chronological split\
-   No look-ahead bias

### Trading Costs

-   0.125% commission per leg\
-   0.25% annual borrow rate\
-   80% capital deployment

------------------------------------------------------------------------

#  Pair Selection Strategy

1.  Correlation screening (\>0.7)\
2.  Engle--Granger regression + ADF on residuals\
3.  Johansen cointegration test\
4.  Final selection based on correlation + ADF p-value + Johansen
    strength

------------------------------------------------------------------------

#  Kalman Filter Implementation

Two Kalman Filters:

### **KF1 -- Dynamic Hedge Ratio βₜ**

Produces: - Intercept\
- Dynamic hedge ratio\
- Raw spread: `s_t = y_t - β_t x_t`

### \*\*KF2 -- Smoothed Spread `\hat{s}`{=tex}\_t\*\*

Produces stable spread used for signals and ADF rolling validation.

------------------------------------------------------------------------

#  Sequential Decision Analysis (SDA)

The full system is modeled under Powell's SDA:

State:\
\[ S_t = (β_t, `\hat{s}`{=tex}*t, P\^β_t, P\^s_t, spread*{t-20:t},
cap_t, pos_t) \]

Decisions: - open_long\
- open_short\
- close\
- hold

------------------------------------------------------------------------

#  Trading Strategy Logic

Entry:\
- Long if Z \< -1\
- Short if Z \> 1\
- Only if ADF p ≤ 0.05

Exit:\
- \|Z\| \< 0.5

Stop-loss:\
- \|Z\| \> 3.5

Position Size: \[ n_t = `\left`{=tex}`\lfloor `{=tex}rac{0.80
`\cdot `{=tex}cap_t}{\|y_t\| + \|β_t x_t\|} ightfloor \]

------------------------------------------------------------------------

#  Backtesting Requirements

-   Daily hedge update\
-   Smoothed spread update\
-   Commission and borrow integration\
-   Train/Test/Validation evaluation\
-   Full trade logs

------------------------------------------------------------------------

#  Results Summary (GOOGL--HD)

### Train

-   -11.85%\
-   \$881,509 final\
-   Sharpe -0.1627

### Test

-   +53.89%\
-   \$1,538,861\
-   Sharpe 1.128

### Validation

-   +34.98%\
-   \$2,077,107\
-   Sharpe 0.8557

### Combined Out-of-Sample

Strong, robust, resilient performance.

------------------------------------------------------------------------

#  Conclusions

-   The system is theoretically consistent and statistically grounded\
-   Out-of-sample performance is strong\
-   Low win-rate offset by large tail winners\
-   Costs are fully absorbed\
-   Possible improvements: adaptive thresholds, regime detection,
    ML-signal validation

------------------------------------------------------------------------

#  Run

``` bash
python main.py
```
