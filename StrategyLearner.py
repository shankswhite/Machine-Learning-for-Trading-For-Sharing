""""""
import numpy as np

"""  		  	   		 	   			  		 			     			  	 
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   			  		 			     			  	 
Atlanta, Georgia 30332  		  	   		 	   			  		 			     			  	 
All Rights Reserved  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Template code for CS 4646/7646  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   			  		 			     			  	 
works, including solutions to the projects assigned in this course. Students  		  	   		 	   			  		 			     			  	 
and other users of this template code are advised not to share it with others  		  	   		 	   			  		 			     			  	 
or to make it available on publicly viewable websites including repositories  		  	   		 	   			  		 			     			  	 
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   			  		 			     			  	 
or edited.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
We do grant permission to share solutions privately with non-students such  		  	   		 	   			  		 			     			  	 
as potential employers. However, sharing with other current or future  		  	   		 	   			  		 			     			  	 
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   			  		 			     			  	 
GT honor code violation.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
-----do not edit anything above this line---  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
Student Name: Xiaofeng Zhao		  	   		 	   			  		 			     			  	 
GT User ID: xzhao474		  	   		 	   			  		 			     			  	 
GT ID: 903957020		  	   		 	   			  		 			     			  	 
"""  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import datetime as dt  		  	   		 	   			  		 			     			  	 
import random  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import pandas as pd  		  	   		 	   			  		 			     			  	 
from util import get_data, plot_data

import QLearner as ql
from marketsimcode import compute_portvals
import indicators


def author():
    return 'xzhao474'


class StrategyLearner(object):  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		 	   			  		 			     			  	 
        If verbose = False your code should not generate ANY output.  		  	   		 	   			  		 			     			  	 
    :type verbose: bool  		  	   		 	   			  		 			     			  	 
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		 	   			  		 			     			  	 
    :type impact: float  		  	   		 	   			  		 			     			  	 
    :param commission: The commission amount charged, defaults to 0.0  		  	   		 	   			  		 			     			  	 
    :type commission: float  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    # constructor  		  	   		 	   			  		 			     			  	 
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Constructor method  		  	   		 	   			  		 			     			  	 
        """
        self.learner = None
        self.verbose = verbose  		  	   		 	   			  		 			     			  	 
        self.impact = impact  		  	   		 	   			  		 			     			  	 
        self.commission = commission

  		  	   		 	   			  		 			     			  	 
    # this method should create a QLearner, and train it for trading  		  	   		 	   			  		 			     			  	 
    def add_evidence(  		  	   		 	   			  		 			     			  	 
        self,  		  	   		 	   			  		 			     			  	 
        symbol="IBM",  		  	   		 	   			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		 	   			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),  		  	   		 	   			  		 			     			  	 
        sv=10000,  		  	   		 	   			  		 			     			  	 
    ):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Trains your strategy learner over a given time frame.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param symbol: The stock symbol to train on  		  	   		 	   			  		 			     			  	 
        :type symbol: str  		  	   		 	   			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			     			  	 
        :type sd: datetime  		  	   		 	   			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			     			  	 
        :type ed: datetime  		  	   		 	   			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
        :type sv: int  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	

        self.learner = ql.QLearner(num_states=81,
                                   num_actions=3,
                                   alpha=0.2,
                                   gamma=0.9,
                                   rar=0.1,
                                   radr=0.99,
                                   dyna=200,
                                   verbose=False)

        curr_cash = sv
        curr_position = 0

        df_prices = get_data([symbol], pd.date_range(sd, ed))
        if 'SPY' in df_prices.columns and symbol != 'SPY':
            df_prices = df_prices.drop(['SPY'], axis=1)

        # print(df_prices.head())

        df_trades = pd.DataFrame(data=0, index=df_prices.index, columns=[symbol])

        macd = indicators.moving_average_convergence_divergence(symbol, sd, ed, plot=False)

        bbp_window_size = 20
        bbp = indicators.bollinger_bands(symbol, sd - dt.timedelta(days=bbp_window_size + 8), ed,
                                         window_size=bbp_window_size, plot=False)
        bbp = bbp[sd:ed]

        mtm = indicators.momentum(symbol, sd - dt.timedelta(days=bbp_window_size + 8), ed, plot=False)
        mtm = mtm[sd:ed]

        discretized_macd, discretized_bbp, discretized_mtm = get_discretized_indicators(macd, bbp, mtm)

        for i in range(1, len(df_prices)):
            today = df_prices.index[i]
            yesterday = df_prices.index[i - 1]

            current_state = get_state(df_trades.loc[yesterday][symbol], discretized_macd.loc[today],
                                      discretized_bbp.loc[today], discretized_mtm.loc[today])

            reward = (df_prices.loc[today, symbol] - df_prices.loc[yesterday, symbol]) * curr_position

            action = self.learner.query(current_state, reward)

            trade = action_to_trade(action, curr_position)

            if trade != 0:
                trade_impact_cost = abs(trade) * df_prices.loc[today, symbol] * self.impact
                total_cost = abs(trade) * df_prices.loc[today, symbol] + trade_impact_cost + self.commission
                curr_cash -= total_cost
            curr_position += trade

            df_trades.loc[today, symbol] = trade

        if self.verbose:
            print(df_trades)
            # print(df_trades.max().max())
            # pass

  		  	   		 	   			  		 			     			  	 
    # this method should use the existing policy and test it against new data  		  	   		 	   			  		 			     			  	 
    def testPolicy(  		  	   		 	   			  		 			     			  	 
        self,  		  	   		 	   			  		 			     			  	 
        symbol="IBM",  		  	   		 	   			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),  		  	   		 	   			  		 			     			  	 
        ed=dt.datetime(2010, 1, 1),  		  	   		 	   			  		 			     			  	 
        sv=10000,  		  	   		 	   			  		 			     			  	 
    ):  		  	   		 	   			  		 			     			  	 
        """  		  	   		 	   			  		 			     			  	 
        Tests your learner using data outside of the training data  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
        :param symbol: The stock symbol that you trained on on  		  	   		 	   			  		 			     			  	 
        :type symbol: str  		  	   		 	   			  		 			     			  	 
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		 	   			  		 			     			  	 
        :type sd: datetime  		  	   		 	   			  		 			     			  	 
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		 	   			  		 			     			  	 
        :type ed: datetime  		  	   		 	   			  		 			     			  	 
        :param sv: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
        :type sv: int  		  	   		 	   			  		 			     			  	 
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		 	   			  		 			     			  	 
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		 	   			  		 			     			  	 
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		 	   			  		 			     			  	 
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		 	   			  		 			     			  	 
        :rtype: pandas.DataFrame  		  	   		 	   			  		 			     			  	 
        """

        """Tests the learned strategy using new data."""

        df_prices = get_data([symbol], pd.date_range(sd, ed))
        if 'SPY' in df_prices.columns and symbol != 'SPY':
            df_prices.drop(['SPY'], axis=1, inplace=True)

        df_trades = pd.DataFrame(data=0, index=df_prices.index, columns=[symbol])

        macd = indicators.moving_average_convergence_divergence(symbol, sd, ed, plot=False)

        bbp_window_size = 20
        bbp = indicators.bollinger_bands(symbol, sd - dt.timedelta(days=bbp_window_size + 8), ed,
                                         window_size=bbp_window_size, plot=False)
        bbp = bbp[sd:ed]

        mtm = indicators.momentum(symbol, sd - dt.timedelta(days=bbp_window_size + 8), ed, plot=False)
        mtm = mtm[sd:ed]

        discretized_macd, discretized_bbp, discretized_mtm = get_discretized_indicators(macd, bbp, mtm)

        for i in range(len(df_prices)):
            today = df_prices.index[i]
            current_position = df_trades.loc[:today].sum().iloc[0]  # Sum of trades up to today

            current_state = get_state(df_trades.loc[today][symbol], discretized_macd.loc[today],
                                      discretized_bbp.loc[today], discretized_mtm.loc[today])

            action = self.learner.querysetstate(current_state)

            trade = action_to_trade(action, current_position)

            df_trades.loc[today][symbol] = trade

        if self.verbose:
            print(df_trades)
            # print(df_trades.min().min())
            # pass

        return df_trades


def get_discretized_indicators(macd, bbp, mtm):
    # print(macd)

    # 1 long, 0 hold, -1 short
    discretized_macd = macd.squeeze().map(lambda x: 1 if x > 1 else (-1 if x < -1.35 else 0))
    discretized_bbp = bbp.squeeze().map(lambda x: 1 if x > 0.95 else (-1 if x < 0.1 else 0))
    discretized_mtm = mtm.squeeze().map(lambda x: 1 if x > 0.4 else (-1 if x < -0.4 else 0))

    return discretized_macd, discretized_bbp, discretized_mtm


def get_state(position, macd, bbp, mtm):
    if position == -1000:
        position_index = 0
    elif position == 0:
        position_index = 1
    else:
        position_index = 2

    macd_index = macd + 1
    bbp_index = bbp + 1
    mtm_index = mtm + 1

    state = position_index * (3 ** 3) + macd_index * (3 ** 2) + bbp_index * 3 + mtm_index
    return state


def action_to_trade(action, current_position):
    expected_trade = 0
    if action == 0:
        expected_trade = -1000 - current_position
    elif action == 2:
        expected_trade = 1000 - current_position
    trade = max(-1000 - current_position, min(expected_trade, 1000 - current_position))
    return trade


if __name__ == "__main__":
    # print("One does not simply think up a strategy")
    symbol = "JPM"
    learner = StrategyLearner(verbose=True, impact=0.000)  # constructor
    learner.add_evidence(symbol=symbol, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df_trades = learner.testPolicy(symbol=symbol, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
    # print(df_trades)