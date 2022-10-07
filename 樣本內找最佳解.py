import os
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import time
import requests
import datetime as dt
path=os.getcwd()
print(path)
data = pd.read_csv(path+'/BTC_PERP.csv', parse_dates=True, index_col='startTime')
funding = pd.read_csv(path+'/BTC_funding1.csv', parse_dates=True, index_col='time')
print(data)
print(funding)

df_inSample=data.loc[:'2022-2']
df_outSample=data.loc['2022-3':]

rule='1H'
df_hour=data.resample(rule=rule,closed='left',label='left').first()[['open']]
df_funding=pd.concat([df_hour,funding],axis=1)
df_funding=df_funding.fillna(method='bfill',limit=2).fillna(0)
print(df_funding)


optimizationList=[]
fund=100
money=100
feeRate=0.003
#lenght=40
#stdTimes=2
rule='1D'


for length in range(10,110,10):
    for stdTimes in np.arange(0.5,3,0.5):
        stdTimes=np.round(stdTimes,1)
        print('---------')
        print(f'length:{length}')
        print(f'stdTimes:{stdTimes}')
        d1 = df_inSample.resample(rule=rule, closed='left', label='left').first()[['open']]
        d2 = df_inSample.resample(rule=rule, closed='left', label='left').max()[['high']]
        d3 = df_inSample.resample(rule=rule, closed='left', label='left').min()[['low']]
        d4 = df_inSample.resample(rule=rule, closed='left', label='left').last()[['close']]
        d5 = df_inSample.resample(rule=rule, closed='left', label='left').sum()[['volume']]
        df = pd.concat([d1, d2, d3, d4, d5], axis=1)

        df['ma'] = df['close'].rolling(window=length, center=False).mean()
        df['std'] = df['close'].rolling(window=length, center=False).std()

        BS = None
        buy = []
        sell = []
        sellshort = []
        buytocover = []
        profit_list = [0]
        profit_fee_list = [0]
        profit_fee_list_realized = []


        def fundingPayment(df_funding, side, unit, t1, t2):
            if len(df_funding.loc[t1:t2]) == 0:
                fee = 0
            else:
                fr = np.array(df_funding.loc[t1:t2])
                fee = unit * np.dot(fr[:, 0], fr[:, 1])
            if side == 'long':
                return -fee
            elif side == 'short':
                return fee

        for i in range(len(df)):
            if i == len(df) - 1:
                break
            if BS == None:
                profit_list.append(0)
                profit_fee_list.append(0)
                if df['close'][i] > df['ma'][i] + stdTimes * df['std'][i]:
                    tempSize = money / df['open'][i + 1]
                    BS = 'B'
                    t = i + 1
                    buy.append(t)
                    t1 = df.index[i + 1]
                elif df['close'][i] < df['ma'][i] - stdTimes * df['std'][i]:
                    tempSize = money / df['open'][i + 1]
                    BS = 'S'
                    t = i + 1
                    sellshort.append(t)
                    t1 = df.index[i + 1]

            elif BS == 'B':
                profit = tempSize * (df['open'][i + 1] - df['open'][i])
                profit_list.append(profit)
                t2 = df.index[i + 1]
                fundingFee = fundingPayment(df_funding, 'long', tempSize, df.index[t], t2)

                if df['close'][i] <= df['ma'][i] or i == len(df) - 2:
                    pl_round = tempSize * (df['open'][i + 1] - df['open'][t])
                    profit_fee = profit - money * feeRate - (money + pl_round) * feeRate + fundingFee
                    profit_fee_list.append(profit_fee)
                    sell.append(i + 1)
                    BS = None

                    profit_fee_realized = pl_round - money * feeRate - (money + pl_round) * feeRate + fundingFee
                    profit_fee_list_realized.append(profit_fee_realized)

                else:
                    profit_fee = profit
                    profit_fee_list.append(profit)
                    t1 = df.index[i + 1]

            elif BS == 'S':
                profit = tempSize * (df['open'][i] - df['open'][i + 1])
                profit_list.append(profit)
                t2 = df.index[i + 1]
                fundingFee = fundingPayment(df_funding, 'short', tempSize, df.index[t], t2)

                if df['close'][i] >= df['ma'][i] or i == len(df) - 2:
                    pl_round = tempSize * (df['open'][t] - df['open'][i + 1])
                    profit_fee = profit - money * feeRate - (money + pl_round) * feeRate + fundingFee
                    profit_fee_list.append(profit_fee)
                    buytocover.append(i + 1)
                    BS = None

                    profit_fee_realized = pl_round - money * feeRate - (money + pl_round) * feeRate + fundingFee
                    profit_fee_list_realized.append(profit_fee_realized)
                else:
                    profit_fee = profit
                    profit_fee_list.append(profit_fee)
                    t1 = df.index[i + 1]
        equity = pd.DataFrame({'profit': np.cumsum(profit_list), 'profitfee': np.cumsum(profit_fee_list)},
                              index=df.index)
        equity['equity'] = equity['profitfee'] + fund
        equity['drawdown_percent'] = (equity['equity'] / equity['equity'].cummax()) - 1
        equity['drawdown'] = equity['equity'] - equity['equity'].cummax()
        ret = equity['equity'][-1] / equity['equity'][0] - 1
        mdd = abs(equity['drawdown_percent'].min())
        calmarRatio = ret / mdd

        optimizationList.append([length, stdTimes, ret, calmarRatio])


optResult=pd.DataFrame(optimizationList,columns=['length','stdTimes','ret','calmarRatio'])
print(optResult)

print(optResult[optResult['stdTimes']==2].sort_values('ret',ascending=False))

pic = optResult.pivot('length', 'stdTimes', 'ret')
sns.heatmap(data = pic).set(title='Return')
plt.show()

pic = optResult.pivot('length', 'stdTimes', 'calmarRatio')
sns.heatmap(data = pic).set(title='Calmar Ratio')
plt.show()


