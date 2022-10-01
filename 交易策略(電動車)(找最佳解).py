import os
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import time
import requests
import pandas_datareader as data
from datetime import datetime

df=pd.read_csv('台達電資料.csv', parse_dates=True)
df['Date'] = pd.to_datetime(df['Date'])
print(df)

BS=None
optimizationList=[]
buy=[]
sell=[]
sellshort=[]
buytocover=[]
profit_list=[0]
profit_fee_list=[0]
profit_fee_list_realized=[]
fund=100
money=100
feerate=0.001425
tax=0.003
#length=20
#stdTimes=1


for length in range(10,110,10):
    for stdTimes in np.arange(0.5,3,0.5):
        print('---------')
        print(f'length:{length}')
        print(f'stdTimes:{stdTimes}')
        df['ma']=df['Close'].rolling(window=length,center=False).mean()
        df['std'] = df['Close'].rolling(window=length, center=False).std()
        BS = None
        buy = []
        sell = []
        sellshort = []
        buytocover = []
        profit_list = [0]
        profit_fee_list = [0]
        profit_fee_list_realized = []
        for i in range(len(df)):
            if i == len(df) - 1:
                break
            if BS == None:
                profit_list.append(0)
                profit_fee_list.append(0)
                if df['Close'][i] > df['ma'][i] + df['std'][i] * stdTimes:
                    tempSize = money / df['Open'][i + 1]
                    BS = 'B'
                    t = i + 1
                    buy.append(t)
                    t1 = df['Date'][i + 1]
                elif df['Close'][i] < df['ma'][i] - df['std'][i] * stdTimes:
                    tempSize = money / df['Open'][i + 1]
                    BS = 'S'
                    t = i + 1
                    sellshort.append(t)
                    t1 = df['Date'][i + 1]
            elif BS == 'B':
                profit = tempSize * (df['Open'][i + 1] - df['Open'][i])
                profit_list.append(profit)
                t2 = df['Date'][i + 1]

                if df['Close'][i] <= df['ma'][i] or i == len(df) - 2:
                    pl_round = tempSize * (df['Open'][i + 1] - df['Open'][i + 1])
                    profit_fee = profit - money * feerate - (money + pl_round) * tax
                    profit_fee_list.append(profit_fee)
                    sell.append(i + 1)
                    BS = None
                    profit_fee_realized = pl_round - money * feerate - (money + pl_round) * tax
                    profit_fee_list_realized.append(profit_fee_realized)

                else:
                    profit_fee = profit
                    profit_fee_list.append(profit_fee)
                    t1 = df['Date'][i + 1]

            elif BS == 'S':
                profit = tempSize * (df['Open'][i] - df['Open'][i + 1])
                profit_list.append(profit)
                t2 = df['Date'][i + 1]

                if df['Close'][i] >= df['ma'][i] or i == len(df) - 2:
                    pl_round = tempSize * (df['Open'][t] - df['Open'][i + 1])
                    profit_fee = profit - money * feerate - (money + pl_round) * tax
                    profit_fee_list.append(profit_fee)
                    buytocover.append(i + 1)
                    BS = None

                    profit_fee_realized = pl_round - money * feerate - (money + pl_round) * tax
                    profit_fee_list_realized.append(profit_fee_realized)

                else:
                    profit_fee = profit
                    profit_fee_list.append(profit_fee)
                    t1 = df['Date'][i + 1]
        equity = pd.DataFrame({'profit': np.cumsum(profit_list), 'profitfee': np.cumsum(profit_fee_list)},
                              index=df['Date'])
        equity['equity'] = equity['profitfee'] + fund
        equity['drawdown_percent'] = (equity['equity'] / equity['equity'].cummax()) - 1
        equity['drawdown'] = equity['equity'] - equity['equity'].cummax()
        ret = equity['equity'][-1] / equity['equity'][0] - 1
        mdd = abs(equity['drawdown_percent'].min())
        calmarRatio = ret / mdd

        optimizationList.append([length, stdTimes, ret, calmarRatio])


optResult=pd.DataFrame(optimizationList,columns=['length','stdTimes','ret','calmarRatio'])
print(optResult)

print(optResult[optResult['stdTimes']==0.5].sort_values('ret',ascending=False))

pic = optResult.pivot('length', 'stdTimes', 'ret')
sns.heatmap(data = pic).set(title='Return')
plt.show()

pic = optResult.pivot('length', 'stdTimes', 'calmarRatio')
sns.heatmap(data = pic).set(title='Calmar Ratio')
plt.show()