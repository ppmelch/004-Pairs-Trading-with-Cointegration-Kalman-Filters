from data_processing import clean_data, dataset_split
from visualization import plot_spread, plot_normalized_data

from backtesting import backtest

tickers = ['MSFT', 'TMO']

def main():

    data = clean_data(tickers, intervalo="15y")
    train, test, validation = dataset_split(data)



    plot_normalized_data(train)
    plot_spread(train)


if __name__ == "__main__":
    main()

