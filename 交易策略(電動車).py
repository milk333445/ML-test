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
length=50
stdTimes=0.5
df['ma']=df['Close'].rolling(window=length,center=False).mean()
df['std'] = df['Close'].rolling(window=length, center=False).std()
print(df)
for i in range(len(df)):
    if i==len(df)-1:
        break
    if BS==None:
        profit_list.append(0)
        profit_fee_list.append(0)
        if df['Close'][i] > df['ma'][i]+df['std'][i]*stdTimes:
            tempSize=money/df['Open'][i+1]
            BS='B'
            t=i+1
            buy.append(t)
            t1=df['Date'][i+1]
        elif df['Close'][i] < df['ma'][i]-df['std'][i]*stdTimes:
            tempSize = money / df['Open'][i + 1]
            BS = 'S'
            t = i + 1
            sellshort.append(t)
            t1 = df['Date'][i + 1]
    elif BS=='B':
        profit=tempSize*(df['Open'][i+1]-df['Open'][i])
        profit_list.append(profit)
        t2=df['Date'][i+1]

        if df['Close'][i] <= df['ma'][i] or i==len(df)-2:
            pl_round=tempSize*(df['Open'][i+1]-df['Open'][i+1])
            profit_fee=profit-money*feerate-(money+pl_round)*tax
            profit_fee_list.append(profit_fee)
            sell.append(i+1)
            BS=None
            profit_fee_realized= pl_round-money*feerate-(money+pl_round)*tax
            profit_fee_list_realized.append(profit_fee_realized)

        else:
            profit_fee=profit
            profit_fee_list.append(profit_fee)
            t1=df['Date'][i+1]

    elif BS=='S':
        profit=tempSize*(df['Open'][i]-df['Open'][i+1])
        profit_list.append(profit)
        t2=df['Date'][i+1]

        if df['Close'][i] >= df['ma'][i] or i==len(df)-2:
            pl_round = tempSize * (df['Open'][t] - df['Open'][i + 1])
            profit_fee = profit - money * feerate - (money + pl_round) * tax
            profit_fee_list.append(profit_fee)
            buytocover.append(i + 1)
            BS = None

            profit_fee_realized = pl_round-money*feerate-(money+pl_round)*tax
            profit_fee_list_realized.append(profit_fee_realized)

        else:
            profit_fee = profit
            profit_fee_list.append(profit_fee)
            t1 = df['Date'][i + 1]


print(profit_list)
print(profit_fee_list)
print(df['Date'])

equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=df['Date'])
print(equity)
equity.plot(grid=True, figsize=(12, 6))
plt.show()

print(buy)
print(sell)
print(sellshort)
print(buytocover)

equity['equity']=equity['profitfee']+fund
equity['drawdown_percent']=(equity['equity']/equity['equity'].cummax())-1
equity['drawdown'] = equity['equity'] - equity['equity'].cummax()
print(equity)

fig, ax = plt.subplots(figsize = (16,6))

high_index = equity[equity['profitfee'].cummax() == equity['profitfee']].index
equity['profitfee'].plot(label = 'Total Profit', ax = ax, c = 'r', grid=True)
plt.fill_between(equity['drawdown'].index, equity['drawdown'], 0, facecolor  = 'r', label = 'Drawdown', alpha=0.5)
plt.scatter(high_index, equity['profitfee'].loc[high_index],c = '#02ff0f', label = 'High')

plt.legend()
plt.ylabel('Accumulated Profit')
plt.xlabel('Time')
plt.title('Profit & Drawdown',fontsize  = 16)
plt.show()


fig,ax=plt.subplots(figsize=(16,6))
df['Close'].plot(label='Close Price',ax=ax,c='gray',grid=True,alpha=0.8)
plt.scatter(df['Close'].iloc[buy].index,df['Close'].iloc[buy],c = 'orangered', label = 'Buy', marker='^', s=60)
plt.scatter(df['Close'].iloc[sell].index, df['Close'].iloc[sell],c = 'orangered', label = 'Sell', marker='v', s=60)
plt.scatter(df['Close'].iloc[sellshort].index, df['Close'].iloc[sellshort],c = 'limegreen', label = 'Sellshort', marker='v', s=60)
plt.scatter(df['Close'].iloc[buytocover].index, df['Close'].iloc[buytocover],c = 'limegreen', label = 'Buytocover', marker='^', s=60)

plt.legend()
plt.ylabel('stock price')
plt.xlabel('Time')
plt.title('Price Movement',fontsize  = 16)
plt.show()

print(profit_fee_list_realized)


profit=equity['profitfee'].iloc[-1]
ret=equity['equity'][-1]/equity['equity'][0]-1
mdd=abs(equity['drawdown_percent'].min())
calmarRatio = ret / mdd
tradeTimes = len(buy)+len(sellshort)
winRate=len([i for i in profit_fee_list_realized if i >0])/len(profit_fee_list_realized)
profitFactor=sum([i for i in profit_fee_list_realized if i >0])/abs(sum([i for i in profit_fee_list_realized if i < 0]))
winLossRatio=np.mean([i for i in profit_fee_list_realized if i > 0]) / abs(np.mean([i for i in profit_fee_list_realized if i < 0]))
print(f'profit: ${np.round(profit,2)}')
print(f'return: {np.round(ret,4)*100}%')
print(f'mdd: {np.round(mdd,4)*100}%')
print(f'calmarRatio: {np.round(calmarRatio,2)}')
print(f'tradeTimes: {tradeTimes}')
print(f'winRate: {np.round(winRate,4)*100}%')
print(f'profitFactor: {np.round(profitFactor,2)}')
print(f'winLossRatio: {np.round(winLossRatio,2)}')
