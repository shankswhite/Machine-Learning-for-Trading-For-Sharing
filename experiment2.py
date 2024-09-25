import numpy as np
import pandas as pd

from StrategyLearner import StrategyLearner
from ManualStrategy import ManualStrategy
from marketsimcode import compute_portvals
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data, plot_data


def author():
    return 'xz'


def experiment2():
    sv = 100000
    symbol = 'JPM'

    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)

    manual = ManualStrategy()

    print("Impact: 0")
    learner1 = StrategyLearner(verbose=False, impact=0, commission=0)
    learner1.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner1_trades = learner1.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner1_portval = compute_portvals(learner1_trades, start_val=sv, commission=0, impact=0.000)


    print("\nImpact: 0.005")
    learner2 = StrategyLearner(verbose=False, impact=0.005, commission=0)
    learner2.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner2_trades = learner2.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner2_portval = compute_portvals(learner2_trades, start_val=sv, commission=0, impact=0.000)

    print("\nImpact: 0.01")
    learner3 = StrategyLearner(verbose=False, impact=0.01, commission=0)
    learner3.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner3_trades = learner3.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    learner3_portval = compute_portvals(learner3_trades, start_val=sv, commission=0, impact=0.000)

    plot1(learner1_portval, learner2_portval, learner3_portval, "Daily Return")
    plot2(learner1_portval, learner2_portval, learner3_portval, "Sharp Ratio")


def plot1(strategy1, strategy2, strategy3, label):
    strategy1_normalized = strategy1 / strategy1.iloc[0]
    strategy2_normalized = strategy2 / strategy2.iloc[0]
    strategy3_normalized = strategy3 / strategy3.iloc[0]

    plt.figure(figsize=(10, 6))
    plt.plot(strategy1_normalized, label="Impact: 0", color="green")
    plt.plot(strategy2_normalized, label="Impact: 0.005", color="red")
    plt.plot(strategy3_normalized, label="Impact: 0.01", color="blue")

    plt.title(f"Impact Effect on {label.capitalize()}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.xticks(rotation=30)
    plt.legend(loc='best')
    plt.grid(visible=True)
    plt.savefig(f"./images/experiment2_{label.capitalize()}.png", bbox_inches='tight')
    # plt.show()
    plt.clf()


def calculate_sharpe_ratio(portvals):
    daily_returns = portvals.pct_change(1)[1:]  # Calculate daily returns, skip the first NaN
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)  # Assuming 252 trading days in a year
    return sharpe_ratio


def plot2(strategy1, strategy2, strategy3, label):
    sharpe_ratio_strategy1 = calculate_sharpe_ratio(strategy1)
    sharpe_ratio_strategy2 = calculate_sharpe_ratio(strategy2)
    sharpe_ratio_strategy3 = calculate_sharpe_ratio(strategy3)

    strategies = ['Impact: 0', 'Impact: 0.005', 'Impact: 0.01']
    sharpe_ratios = [sharpe_ratio_strategy1, sharpe_ratio_strategy2, sharpe_ratio_strategy3]

    plt.figure(figsize=(10, 6))
    plt.bar(strategies, sharpe_ratios, color=['green', 'red', 'blue'])

    plt.title(f"Sharpe Ratios by Impact Level for {label}")
    plt.ylabel("Sharpe Ratio")
    plt.grid(axis='y', linestyle='--')

    for i in range(len(strategies)):
        plt.text(i, sharpe_ratios[i], f"{sharpe_ratios[i]:.2f}", ha='center', va='bottom')

    plt.savefig(f"./images/experiment2_sharpe_{label}.png", bbox_inches='tight')
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
    experiment2()
