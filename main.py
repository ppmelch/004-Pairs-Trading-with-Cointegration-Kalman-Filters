from prints import backtest_pair_splits
from cointegration import selected_pair
from data_processing import clean_data, dataset_split

tickers = ['GOOGL', 'HD']


def main():
    """
    Run the full workflow: clean historical data, split it into partitions,
    extract the target pair, and execute the backtest across train/test/validation.
    """
    data = clean_data(tickers, "15y")
    train, test, val = dataset_split(data)
    pair_train = selected_pair(train, tickers[0], tickers[1])
    pair_test = selected_pair(test, tickers[0], tickers[1])
    pair_val = selected_pair(val, tickers[0], tickers[1])
    backtest_pair_splits(pair_train, pair_test, pair_val)


if __name__ == "__main__":
    main()
