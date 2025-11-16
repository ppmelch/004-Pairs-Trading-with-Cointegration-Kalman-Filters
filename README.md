# 🧠 004 Pairs Trading with Cointegration & Kalman Filters
### ITESO — Market Microstructure and Trading Systems  
**Autores:**  
- José Armando Melchor Soto  
- Rolando Fortanell Canedo  

------------------------------------------------------------------------

# 📥 Installation & Requirements

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

# 📌 Project Overview

This project implements a **statistical arbitrage pairs trading
strategy** using:

-   Correlation, Engle--Granger & Johansen cointegration\
-   Two Kalman Filters (dynamic hedge ratio + smoothed spread)\
-   Z-score mean reversion\
-   Sequential Decision Analysis (SDA)\
-   Realistic backtesting (commissions, borrow, slippage)

The system updates the hedge ratio and spread through Kalman filters,
validates cointegration in real-time, and makes daily decisions to
open/close positions.

------------------------------------------------------------------------

# 🧩 System Architecture

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
    ├── rquirements.txt
    └── README.md
    

------------------------------------------------------------------------

# 🔍 Methodology

## 1. Data Cleaning

15 years of daily prices downloaded via yfinance and aligned.

## 2. Pair Selection

### ✓ Correlation (\> 0.7)

### ✓ Engle--Granger (OLS + ADF)

Cointegration if ADF p-value \< 0.05.

### ✓ Johansen Test

Uses trace statistic and eigenvectors to confirm long‑run equilibrium.

Pairs ranked by:

1.  ADF p-value\
2.  Johansen strength\
3.  Correlation

------------------------------------------------------------------------

# ⚙️ Kalman Filters + SDA

## Kalman Filter 1 --- Dynamic Hedge Ratio βₜ

Updates intercept & slope daily:\
\[ s_t = y_t - `\beta`{=tex}\_t x_t \]

## Kalman Filter 2 --- Smoothed Spread ( `\hat{s}`{=tex}\_t  )

Removes noise and is used for Z-score and rolling ADF.

## System State

\[ S_t = (`\beta`{=tex}\_t, `\hat{s}`{=tex}\_t, P\^`\beta`{=tex}*t,
P\^s_t, spread*{t-20:t}, cap_t, pos_t) \]

------------------------------------------------------------------------

# 🧮 Trading Logic

### Entry

-   Short if Z \> 1.0\
-   Long if Z \< --1.0\
-   Only if rolling ADF \< 0.05

### Exit

-   \|Z\| \< 0.5

### Stop-loss

-   \|Z\| \> 3.5

### Costs

-   0.125% per leg\
-   Borrow BR/252

### Capital

-   Uses 80% of capital, market‑neutral\
-   Share sizing respects βₜ

------------------------------------------------------------------------

# 📊 Backtesting Setup

Dataset split:

-   60% Train\
-   20% Test\
-   20% Validation

Daily loop:

1.  Update Kalman filters\
2.  Compute spread & z-score\
3.  Validate cointegration\
4.  SDA decision\
5.  Apply costs\
6.  Update equity

------------------------------------------------------------------------

# 📈 Results

## TRAIN (60%)

-   Final capital: **\$881,509.97**
-   Return: **--11.85%**
-   Sharpe: **--0.1627**
-   Sortino: **--0.0723**
-   Max Drawdown: **25.6%**
-   Win Rate: **4.29%**

------------------------------------------------------------------------

## TEST (20%)

-   Final capital: **\$1,538,861**
-   Return: **53.89%**
-   Sharpe: **1.128**
-   Sortino: **0.7206**
-   Max Drawdown: **8.78%**
-   Profit: **\$582,116**

------------------------------------------------------------------------

## VALIDATION (20%)

-   Final capital: **\$2,077,107**
-   Return: **34.98%**
-   Sharpe: **0.8557**
-   Sortino: **0.9342**
-   Max Drawdown: **23.94%**
-   Profit: **\$585,169**

------------------------------------------------------------------------

# 🚀 Combined Out‑of‑Sample (Test + Validation)

Strong upward equity curve, resilience, and robust generalization.

------------------------------------------------------------------------

# 🧾 Conclusions

-   The strategy becomes profitable after training.\
-   GOOGL--HD exhibits extremely strong cointegration.\
-   Low win rate but large winners dominate losses.\
-   Robust in out‑of‑sample performance.\
-   Improvements: adaptive thresholds, regime detection, more pairs, ML
    validation.

------------------------------------------------------------------------
