		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import datetime as dt  		  	   		 	   			  		 			     			  	 
import os  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import numpy as np  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
import pandas as pd  		  	   		 	   			  		 			     			  	 
from util import get_data, plot_data
import matplotlib.pyplot as plt


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "xz"


def compute_portvals(
    df_orders,
    start_val=1000000,
    commission=9.95,
    impact=0.005,
):
    """  		  	   		 	   			  		 			     			  	 
    Computes the portfolio values.  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
    :param orders_file: Path of the order file or the file object  		  	   		 	   			  		 			     			  	 
    :type orders_file: str or file object  		  	   		 	   			  		 			     			  	 
    :param start_val: The starting value of the portfolio  		  	   		 	   			  		 			     			  	 
    :type start_val: int  		  	   		 	   			  		 			     			  	 
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		 	   			  		 			     			  	 
    :type commission: float  		  	   		 	   			  		 			     			  	 
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		 	   			  		 			     			  	 
    :type impact: float  		  	   		 	   			  		 			     			  	 
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		 	   			  		 			     			  	 
    :rtype: pandas.DataFrame  		  	   		 	   			  		 			     			  	 
    """  		  	   		 	   			  		 			     			  	 
    # this is the function the autograder will call to test your code  		  	   		 	   			  		 			     			  	 
    # NOTE: orders_file may be a string, or it may be a file object. Your  		  	   		 	   			  		 			     			  	 
    # code should work correctly with either input  		  	   		 	   			  		 			     			  	 
    # TODO: Your code here
    df = df_orders.sort_index()
    start_date = df.index[0]
    end_date = df.index[-1]
    dates = pd.date_range(start_date, end_date)
    syms = df.columns.tolist()
    df_prices = get_data(syms, dates)
    df_prices['CASH'] = 1

    df_trades = df_prices.copy()
    df_trades[:] = 0

    df['Order'] = np.where(df[syms] >= 0, 'BUY', 'SELL')
    df[syms] = df[syms].abs()
    df['Symbol'] = syms[0]
    df.rename(columns={syms[0]: 'Shares'}, inplace=True)


    for index, row in df.iterrows():
        if row['Order'] == "BUY":
            df_trades.loc[index, row['Symbol']] += row['Shares']
            df_trades.loc[index, 'CASH'] += - row['Shares'] * df_prices.loc[index, row['Symbol']]
        elif row['Order'] == "SELL":
            df_trades.loc[index, row['Symbol']] += -row['Shares']
            df_trades.loc[index, 'CASH'] += row['Shares'] * df_prices.loc[index, row['Symbol']]

    df_trades_impact = df_prices.copy()
    df_trades_impact[:] = 0

    for index, row in df.iterrows():
        if row['Order'] == "BUY":
            df_trades_impact.loc[index, row['Symbol']] += row['Shares']
            df_trades_impact.loc[index, 'CASH'] += - row['Shares'] * df_prices.loc[index, row['Symbol']] * (1 + impact)
        elif row['Order'] == "SELL":
            df_trades_impact.loc[index, row['Symbol']] += -row['Shares']
            df_trades_impact.loc[index, 'CASH'] += row['Shares'] * df_prices.loc[index, row['Symbol']] * (1 - impact)

    df_trades_commission = df_trades_impact.copy()
    # print(df_trades_commission)
    for index, row in df.iterrows():
        df_trades_commission.loc[index, 'CASH'] -= commission

    # print(df_trades_commission)

    df_holdings = df_trades.copy()
    df_holdings.iloc[0, -1] = start_val + df_trades_commission.iloc[0, -1]
    df_holdings.iloc[0, :-1] = df_trades_commission.iloc[0, :-1]

    # print(df_trades_commission)
    for i in range(1, df_holdings.shape[0]):
        df_holdings.iloc[i, :] = df_trades_commission.iloc[i, :] + df_holdings.iloc[i - 1, :]

    # print(df_holdings)

    df_value = df_holdings.multiply(df_prices)
    # print(df_value)
    portvals = df_value.sum(axis=1)
  		  	   		 	   			  		 			     			  	 
    # In the template, instead of computing the value of the portfolio, we just  		  	   		 	   			  		 			     			  	 
    # read in the value of IBM over 6 months

    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    # portvals = get_data(["IBM"], pd.date_range(start_date, end_date))
    # portvals = portvals[["IBM"]]  # remove SPY
    # rv = pd.DataFrame(index=portvals.index, data=portvals.values)
  	#
    # return rv
    # print(portvals)
    return portvals

  		  	   		 	   			  		 			     			  	 
  		  	   		 	   			  		 			     			  	 
if __name__ == "__main__":  		  	   		 	   			  		 			     			  	 
    of = tos.testPolicy(symbol = "JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009,12,31), sv = 100000)
    sv = 100000
    # print(of)
    # # Process orders
    df = compute_portvals(of, start_val=sv)
    # print(df)




