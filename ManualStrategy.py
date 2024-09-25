import datetime as dt

import numpy as np
import pandas as pd
from marketsimcode import compute_portvals
import matplotlib.pyplot as plt
from util import get_data, plot_data
import indicators
from scipy.optimize import minimize


def author():
    return 'xz'


class ManualStrategy:
    def testPolicy(self, symbol, sd, ed, sv):
        df_prices = get_data([symbol], pd.date_range(sd, ed))
        # print(df_prices)
        if 'SPY' in df_prices.columns and symbol != 'SPY':
            df_prices = df_prices.drop(['SPY'], axis=1)

        df_trades = pd.DataFrame(data=0, index=df_prices.index, columns=[symbol])
        current_holding = 0

        macd = indicators.moving_average_convergence_divergence(symbol, sd, ed, plot=False)

        bbp_window_size = 20
        bbp = indicators.bollinger_bands(symbol, sd - dt.timedelta(days=bbp_window_size + 8), ed,
                                         window_size=bbp_window_size, plot=False)
        bbp = bbp[sd:ed]

        # rsi = indicators.relative_strength_index(symbol, sd, ed, plot=False)
        mtm = indicators.momentum(symbol, sd - dt.timedelta(days=bbp_window_size + 8), ed, plot=False)
        mtm = mtm[sd:ed]
        # print(mtm)

        for i in range(len(df_prices) - 1):
            if macd.iloc[i].item() > 1:
                macd_weight = 20
            elif macd.iloc[i].item() < -1.35:
                macd_weight = -20
            else:
                macd_weight = 0

            if bbp.iloc[i] < 0.1:
                bbp_weight = 20
            elif bbp.iloc[i] > 0.95:
                bbp_weight = -20
            else:
                bbp_weight = 0

            if mtm.iloc[i] < -0.4:
                mtm_weight = -20
            elif mtm.iloc[i] > 0.4:
                mtm_weight = 20
            else:
                mtm_weight = 0

            weights_sum = macd_weight + bbp_weight + mtm_weight
            if weights_sum > 0:
                trade = min(1000 - current_holding, 1000)
            elif weights_sum < 0:
                trade = -min(1000 + current_holding, 1000)
            else:
                trade = 0

            df_trades[symbol].iloc[i] = trade
            current_holding += trade

        if current_holding != 0:
            df_trades[symbol].iloc[-1] = -current_holding

        return df_trades


def calculate_benchmark(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):
    df_prices = get_data([symbol], pd.date_range(sd, ed))

    if symbol != 'SPY':
        df_prices = df_prices.drop(['SPY'], axis=1)

    df_trades = pd.DataFrame(data=0, index=df_prices.index, columns=[symbol])
    df_trades.loc[df_trades.index[0]] = 1000

    portvals = compute_portvals(
        df_trades,
        start_val=100000,
        commission=0,
        impact=0,
    )

    return portvals


def calculate_statistics(benchmark, strategy):
    cumulative_return_benchmark = benchmark.iloc[-1] / benchmark.iloc[0] - 1
    cumulative_return_strategy = strategy.iloc[-1] / strategy.iloc[0] - 1

    daily_returns_benchmark = benchmark / benchmark.shift(1) - 1
    daily_returns_strategy = strategy / strategy.shift(1) - 1

    standard_deviation_daily_returns_benchmark = daily_returns_benchmark[1:].std()
    standard_deviation_daily_returns_strategy = daily_returns_strategy[1:].std()

    average_daily_return_benchmark = daily_returns_benchmark[1:].mean()
    average_daily_return_strategy = daily_returns_strategy[1:].mean()

    sharpe_ratio_benchmark = (average_daily_return_benchmark) / standard_deviation_daily_returns_benchmark * np.sqrt(252)
    sharpe_ratio_strategy = (average_daily_return_strategy) / standard_deviation_daily_returns_strategy * np.sqrt(252)

    print("")
    print(f"Strategy Cumulative Return: {cumulative_return_strategy}")
    print(f"Strategy Standard Deviation of Daily Returns: {standard_deviation_daily_returns_strategy}")
    print(f"Strategy Average Daily Return: {average_daily_return_strategy}")
    print(f"Strategy Sharpe Ratio: {sharpe_ratio_strategy}")
    print("")
    print(f"Benchmark Cumulative Return: {cumulative_return_benchmark}")
    print(f"Benchmark Standard Deviation of Daily Returns: {standard_deviation_daily_returns_benchmark}")
    print(f"Benchmark Average Daily Return: {average_daily_return_benchmark}")
    print(f"Benchmark Sharpe Ratio: {sharpe_ratio_benchmark}")


def plot(benchmark_portvals, theoretical_portvals, short, long, label):
    benchmark_normalized = benchmark_portvals / benchmark_portvals.iloc[0]
    theoretical_normalized = theoretical_portvals / theoretical_portvals.iloc[0]

    plt.figure(figsize=(10, 6))
    plt.plot(benchmark_normalized, label="Benchmark", color="purple")
    plt.plot(theoretical_normalized, label="Manual Strategy", color="red")


    for date in short:
        plt.axvline(date, color="black", label="Short Entry" if date == short[0] else "")
    for date in long:
        plt.axvline(date, color="blue", label="Long Entry" if date == long[0] else "")

    plt.title(f"Manual Strategy on {label.capitalize()}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.xticks(rotation=30)
    plt.legend(loc='best')
    plt.grid(visible=True)
    plt.savefig(f"./images/Figure_{label}.png", bbox_inches='tight')
    # plt.show()
    plt.clf()


def mark_signal(df_trades):
    long_entries = []
    short_entries = []
    trade_values = df_trades.iloc[:, 0]  # 获取第一列的值
    for i in range(1, len(trade_values)):
        if trade_values.iloc[i] > 0 and trade_values.iloc[i - 1] <= 0:
            long_entries.append(trade_values.index[i])
        elif trade_values.iloc[i] < 0 and trade_values.iloc[i - 1] >= 0:
            short_entries.append(trade_values.index[i])
    return long_entries, short_entries


def generate_report():
    ms = ManualStrategy()

    sv = 100000
    symbol = 'JPM'

    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)
    print("In-Sample Analysis:")
    df_trades_in = ms.testPolicy(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    in_sample_portvals = compute_portvals(df_trades_in, start_val=sv, commission=9.95, impact=0.005)
    in_sample_benchmark_portvals = calculate_benchmark(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed)
    calculate_statistics(in_sample_benchmark_portvals, in_sample_portvals)
    long_entries_in, short_entries_in = mark_signal(df_trades_in)
    # print("Long Entries:", long_entries_in)
    # print("Short Entries:", short_entries_in)
    plot(in_sample_benchmark_portvals, in_sample_portvals, long_entries_in, short_entries_in, 'in_sample')

    out_of_sample_sd = dt.datetime(2010, 1, 1)
    out_of_sample_ed = dt.datetime(2011, 12, 31)
    print("\nOut-of-Sample Analysis:")
    df_trades_out = ms.testPolicy(symbol=symbol, sd=out_of_sample_sd, ed=out_of_sample_ed, sv=sv)

    out_of_sample_portvals = compute_portvals(df_trades_out, start_val=sv, commission=9.95, impact=0.005)
    out_of_sample_benchmark_portvals = calculate_benchmark(symbol=symbol, sd=out_of_sample_sd, ed=out_of_sample_ed)
    calculate_statistics(out_of_sample_benchmark_portvals, out_of_sample_portvals)
    long_entries_out, short_entries_out = mark_signal(df_trades_out)
    # print("Out-of-Sample Long Entries:", long_entries_out)
    # print("Out-of-Sample Short Entries:", short_entries_out)
    plot(out_of_sample_benchmark_portvals, out_of_sample_portvals, long_entries_out, short_entries_out, 'out_of_sample')


if __name__ == "__main__":
    generate_report()
