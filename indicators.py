import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data


def author():
    """  		  	   		 	   			  		 			     			  	 
    :return: The GT username of the student  		  	   		 	   			  		 			     			  	 
    :rtype: str  		  	   		 	   			  		 			     			  	 
    """
    return "xzhao474"  # replace tb34 with your Georgia Tech username.


def bollinger_bands(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), plot=True, window_size=20):
    df_price = get_data([symbol], pd.date_range(sd, ed))
    df_price = df_price[[symbol]].dropna()

    rolling_mean = df_price[symbol].rolling(window=window_size, min_periods=window_size).mean()
    rolling_std = df_price[symbol].rolling(window=window_size, min_periods=window_size).std()
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    bbp = (df_price[symbol] - lower_band) / (upper_band - lower_band)

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(df_price[symbol], label="Stock Price", color="blue")
        plt.plot(rolling_mean, label="Rolling Mean", color="black")
        plt.plot(upper_band, label="Upper Band", color="red")
        plt.plot(lower_band, label="Lower Band", color="green")

        plt.title(f"Bollinger Bands")
        plt.xlabel("Date")
        plt.ylabel("Price")

        plt.fill_between(df_price.index, lower_band, upper_band, color='gray', alpha=0.1)
        plt.grid(visible=True)
        plt.legend()
        plt.savefig("./images/Figure_BBP.png", bbox_inches='tight')

    return bbp


def relative_strength_index(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), plot=True, lookback=14):
    start_date = sd - dt.timedelta(days=lookback * 2)

    prices = get_data([symbol], pd.date_range(start_date, ed))
    prices = prices[[symbol]]
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

    daily_returns = prices[symbol].pct_change().dropna()

    gains = daily_returns.where(daily_returns > 0, 0.0)
    losses = -daily_returns.where(daily_returns < 0, 0.0)

    avg_gain = gains.rolling(window=lookback, min_periods=lookback).mean()
    avg_loss = losses.rolling(window=lookback, min_periods=lookback).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.loc[sd:ed]
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(rsi, label="RSI", color="purple")
        plt.axhline(y=70, color='red', linestyle='--', label='Overbought Threshold (70)')
        plt.axhline(y=30, color='green', linestyle='--', label='Oversold Threshold (30)')

        plt.title(f"Relative Strength Index (RSI)")
        plt.xlabel("Date")
        plt.ylabel("RSI")

        plt.legend(loc='best')
        plt.grid(visible=True)
        plt.savefig(f"./images/Figure_RSI.png", bbox_inches='tight')
        # plt.show()

    return rsi


def exponential_moving_average(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), plot=True, window_size=20):
    extended_sd = sd - dt.timedelta(days=window_size * 2)

    df_price = get_data([symbol], pd.date_range(extended_sd, ed))
    df_price.fillna(method='ffill', inplace=True)
    df_price.fillna(method='bfill', inplace=True)

    df_ema = df_price.ewm(span=window_size, adjust=False).mean().truncate(before=sd)

    normalized_price = df_price[symbol] / df_price[symbol].iloc[0]
    normalized_ema = df_ema[symbol] / df_ema[symbol].iloc[0]

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(normalized_price, label='Normalized Price', color='blue')
        plt.plot(normalized_ema, label='EMA', color='red')

        plt.title(f'Exponential Moving Average (EMA)')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(visible=True)
        plt.savefig("./images/Figure_EMA.png", bbox_inches='tight')
        # plt.show()

    return normalized_ema


def momentum(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), lookback=14, plot=True):

    df_price = get_data([symbol], pd.date_range(sd, ed))
    df_price = df_price[[symbol]].dropna().astype(float)  # Ensure data is numeric and drop NA

    mtm = df_price[symbol] / df_price[symbol].shift(lookback) - 1

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(mtm, label="Momentum", color="purple")

        plt.title("Momentum")
        plt.xlabel("Date")
        plt.ylabel("Momentum")

        plt.axhline(y=0, color='black', linestyle='--')
        plt.legend(loc='best')
        plt.grid(visible=True)
        plt.savefig("./images/Figure_MTM.png", bbox_inches='tight')
        # plt.show()

    return mtm


def moving_average_convergence_divergence(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), plot=True):

    extended_start = sd - dt.timedelta(days=26*2)  # Extend to ensure sufficient data for EMA calculation

    prices = get_data([symbol], pd.date_range(extended_start, ed))
    prices = prices[[symbol]].fillna(method='ffill').fillna(method='bfill')

    ema_short = prices.ewm(span=12, adjust=False).mean()
    ema_long = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    macd_line = macd_line.loc[sd:ed]
    signal_line = signal_line.loc[sd:ed]

    result = macd_line - signal_line
    result = result.loc[sd:ed]

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(macd_line.index, macd_line, label='MACD Line', color='blue')
        plt.plot(signal_line.index, signal_line, label='Signal Line', color='orange')

        plt.axhline(0, linestyle='--', color='red')
        plt.title(f'MACD')
        plt.xlabel('Date')
        plt.ylabel('MACD')
        plt.legend()
        plt.grid(True)
        plt.savefig("./images/Figure_MACD.png", bbox_inches='tight')
        # plt.show()

    return result

if (clicked):
