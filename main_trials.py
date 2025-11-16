from data_processing import clean_data, dataset_split
from cointegration import select_pairs

tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD',
    'JPM', 'BAC', 'WFC', 'C', 'GS',
    'JNJ', 'PFE', 'MRK', 'ABBV', 'TMO',
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',
    'AMZN', 'TSLA', 'HD', 'NKE', 'MCD', "LOW",
    'CAT', 'HON', 'GE', 'BA', 'UPS', "MA", "V"
]


def cointegration():
    """
    Load historical prices, extract the training portion, and compute
    cointegration statistics for all ticker combinations.
    """
    data_pairs = clean_data(tickers, intervalo="15y")
    train, _, _ = dataset_split(data_pairs)
    pairs = select_pairs(train, corr_threshold=0.6, adf_alpha=0.05)
    print("======== SELECTED PAIRS ========")
    print(pairs)


if __name__ == "__main__":
    cointegration()
