import numpy as np
import pandas as pd

from StrategyLearner import StrategyLearner
from ManualStrategy import ManualStrategy
from marketsimcode import compute_portvals
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data, plot_data


def author():
    return 'xzhao474'


def experiment1():
    sv = 100000
    symbol = 'JPM'

    in_sample_sd = dt.datetime(2008, 1, 1)
    in_sample_ed = dt.datetime(2009, 12, 31)

    manual = ManualStrategy()

    print("In-Sample Analysis:")
    manual_trades = manual.testPolicy(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    manual_portval = compute_portvals(manual_trades, start_val=sv, commission=0, impact=0.000)

    learner = StrategyLearner(verbose=False, impact=0.005, commission=9.95)
    learner.add_evidence(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    learner_trades = learner.testPolicy(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed, sv=sv)
    learner_portval = compute_portvals(learner_trades, start_val=sv, commission=0, impact=0.000)

    benchmark_portval = calculate_benchmark(symbol=symbol, sd=in_sample_sd, ed=in_sample_ed)
    calculate_statistics(manual_portval, learner_portval, benchmark_portval)
    plot(manual_portval, learner_portval, benchmark_portval, "In-Sample")
    #
    out_of_sample_sd = dt.datetime(2010, 1, 1)
    out_of_sample_ed = dt.datetime(2011, 12, 31)
    print("\nOut-of-Sample Analysis:")
    manual_trades = manual.testPolicy(symbol=symbol, sd=out_of_sample_sd, ed=out_of_sample_ed, sv=sv)
    manual_portval = compute_portvals(manual_trades, start_val=sv, commission=0, impact=0.000)

    learner = StrategyLearner(verbose=False, impact=0.005, commission=9.95)
    learner.add_evidence(symbol=symbol, sd=out_of_sample_sd, ed=out_of_sample_ed, sv=sv)
    learner_trades = learner.testPolicy(symbol=symbol, sd=out_of_sample_sd, ed=out_of_sample_ed, sv=sv)
    learner_portval = compute_portvals(learner_trades, start_val=sv, commission=0, impact=0.000)

    benchmark_portval = calculate_benchmark(symbol=symbol, sd=out_of_sample_sd, ed=out_of_sample_ed)

    calculate_statistics(manual_portval, learner_portval, benchmark_portval)
    plot(manual_portval, learner_portval, benchmark_portval, "Out-of-Sample")


def calculate_statistics(manual, strategy, benchmark):
    cr_manual = manual.iloc[-1] / manual.iloc[0] - 1
    daily_returns_manual = manual / manual.shift(1) - 1
    sddr_manual = daily_returns_manual[1:].std()
    adr_manual = daily_returns_manual[1:].mean()
    sharpe_ratio_manual = (adr_manual) / sddr_manual * np.sqrt(252)

    cr_strategy = strategy.iloc[-1] / strategy.iloc[0] - 1
    daily_returns_strategy = strategy / strategy.shift(1) - 1
    sddr_strategy = daily_returns_strategy[1:].std()
    adr_strategy = daily_returns_strategy[1:].mean()
    sharpe_ratio_strategy = (adr_strategy) / sddr_strategy * np.sqrt(252)

    cr_benchmark = benchmark.iloc[-1] / benchmark.iloc[0] - 1
    daily_returns_benchmark = benchmark / benchmark.shift(1) - 1
    sddr_benchmark = daily_returns_benchmark[1:].std()
    adr_benchmark = daily_returns_benchmark[1:].mean()
    sharpe_ratio_benchmark = (adr_benchmark) / sddr_benchmark * np.sqrt(252)


    print("Manual Strategy Cumulative Return:", cr_manual)
    print("Manual Strategy Standard Deviation of Daily Returns:", sddr_manual)
    print("Manual Strategy Average Daily Return:", adr_manual)
    print("Manual Strategy Sharpe Ratio:", sharpe_ratio_manual)
    print()

    print("Q Learning Strategy Cumulative Return:", cr_strategy)
    print("Q Learning Strategy Standard Deviation of Daily Returns:", sddr_strategy)
    print("Q Learning Strategy Average Daily Return:", adr_strategy)
    print("Q Learning Strategy Sharpe Ratio:", sharpe_ratio_strategy)
    print()

    print("Benchmark Cumulative Return:", cr_benchmark)
    print("Benchmark Standard Deviation of Daily Returns:", sddr_benchmark)
    print("Benchmark Average Daily Return:", adr_benchmark)
    print("Benchmark Sharpe Ratio:", sharpe_ratio_benchmark)
    print()


def plot(manual, strategy, benchmark, label):
    manual_normalized = manual / manual.iloc[0]
    strategy_normalized = strategy / strategy.iloc[0]
    benchmark_normalized = benchmark / benchmark.iloc[0]

    plt.figure(figsize=(10, 6))
    plt.plot(manual_normalized, label="Manual", color="green")
    plt.plot(strategy_normalized, label="Q Learning", color="red")
    plt.plot(benchmark_normalized, label="benchmark", color="black")

    plt.title(f"Manual Strategy vs Q Learning Strategy on {label.capitalize()}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.xticks(rotation=30)
    plt.legend(loc='best')
    plt.grid(visible=True)
    plt.savefig(f"./images/experiment1_{label}.png", bbox_inches='tight')
    # plt.show()
    plt.clf()


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


if __name__ == "__main__":
    experiment1()
