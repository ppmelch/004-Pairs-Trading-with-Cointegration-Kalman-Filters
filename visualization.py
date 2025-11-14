from libraries import *
from cointegration import OLS


def plot_normalized_data(data: pd.DataFrame) -> None:
    norm_pair = (data - data.mean()) / data.std()

    plt.figure()
    plt.plot(norm_pair.iloc[:, 0], label=norm_pair.columns[0], color=colors[1])
    plt.plot(norm_pair.iloc[:, 1],
             label=norm_pair.columns[1], color=colors[2])
    plt.title('Normalized Price Series')
    plt.xlabel('Date')
    plt.ylabel('Normalized Price')
    plt.legend()
    plt.show()


def plot_spread(data: pd.DataFrame) -> None:
    data = data.copy()

    y = data.iloc[:, 0]
    x = sm.add_constant(data.iloc[:, 1])

    model = sm.OLS(y, x).fit()
    residuals = model.resid
    mean_resid = residuals.mean()

    plt.figure()
    plt.plot(residuals, color=colors[0], label='Spread')
    plt.axhline(mean_resid, color='red',
                linestyle='--', label='Mean Spread')
    plt.title(f'Spread -> {data.columns[0]} & {data.columns[1]}')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    plt.legend()
    plt.show()


