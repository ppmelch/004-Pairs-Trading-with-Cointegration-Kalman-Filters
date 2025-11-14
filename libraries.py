# --- Standard library ---
import os
import warnings
import re, datetime as dt
from dataclasses import dataclass

# --- Third-party libraries: Data analysis ---
import ta
import numpy as np
import pandas as pd
import scipy as sp
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller 
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# --- Third-party libraries: Visualization ---
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.dates import relativedelta
from IPython.display import display


# --- Type hints ---
from typing import List

np.random.seed(42)

plt.rcParams['figure.facecolor'] = 'lightgrey'
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'black'


colors = ["cornflowerblue", "indianred", "darkseagreen", "plum", "dimgray"]