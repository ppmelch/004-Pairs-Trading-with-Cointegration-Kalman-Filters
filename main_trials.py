from classes import coint_config
from data_processing import clean_data, dataset_split
from cointegration import select_pairs
from visualization import plot_spread, plot_normalized_data



tickers = [
    # Technology
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD',

    # Finance
    'JPM', 'BAC', 'WFC', 'C', 'GS',

    # Healthcare
    'JNJ', 'PFE', 'MRK', 'ABBV', 'TMO',

    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',

    # Consumer Discretionary
    'AMZN', 'TSLA', 'HD', 'NKE', 'MCD',

    # Industrials
    'CAT', 'HON', 'GE', 'BA', 'UPS'
]

def cointegration():

    data_pairs = clean_data(tickers, intervalo="15y")
    train, test, validation = dataset_split(data_pairs)


    pairs = select_pairs(train, corr_threshold=0.6, adf_alpha=0.05)
    print("======== PARES SELECCIONADOS ========")
    print(pairs)







if __name__ == "__main__":
    cointegration()
