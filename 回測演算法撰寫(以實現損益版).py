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



fund=100
money=100
feeRate=0.003
length=40
stdTimes=2
rule='1D'

d1 = data.resample(rule=rule, closed='left', label='left').first()[['open']]
d2 = data.resample(rule=rule, closed='left', label='left').max()[['high']]
d3 = data.resample(rule=rule, closed='left', label='left').min()[['low']]
d4 = data.resample(rule=rule, closed='left', label='left').last()[['close']]
d5 = data.resample(rule=rule, closed='left', label='left').sum()[['volume']]

df= pd.concat([d1,d2,d3,d4,d5], axis=1)
#print(df)

df['ma']=df['close'].rolling(window=length,center=False).mean()
df['std'] = df['close'].rolling(window=length, center=False).std()


first=0
BS=None
buy=[]
sell=[]
sellshort=[]
buytocover=[]
timeList=[]
profit_list=[0]
profit_fee_list=[0]


for i in range(len(df)):
    if i==len(df)-1:
        break
    if df['close'][i] > df['ma'][i]+stdTimes*df['std'][i] and BS==None:
        temp=df['open'][i+1]
        tempSize=money/temp
        BS='B'
        t=i+1
        buy.append(t)

        if first==0:
            timeList.append(df.index[i+1])
            first=1
        continue

    if df['close'][i] < df['ma'][i]-stdTimes*df['std'][i] and BS==None:
        temp=df['open'][i+1]
        tempSize=money/temp
        BS='S'
        t=i+1
        sellshort.append(t)
        if first==0:
            timeList.append(df.index[i+1])
            first=1
        continue
#做多出場
    if (df['close'][i] <= df['ma'][i] or i==len(df)-2) and BS=='B':
        profit=tempSize*(df['open'][i+1]-temp)
        profit_fee=profit-money*feeRate-(money+profit)*feeRate
        profit_fee_list.append(profit_fee)
        profit_list.append(profit)
        sell.append(i+1)
        timeList.append(df.index[i])
        BS=None
        continue

#做空出場
    if (df['close'][i]>= df['ma'][i] or i==len(df)-2) and BS=='S':
        profit = tempSize * (temp - df['open'][i + 1])
        profit_fee = profit - money * feeRate - (money + profit) * feeRate
        profit_fee_list.append(profit_fee)
        profit_list.append(profit)
        buytocover.append(i + 1)
        timeList.append(df.index[i])
        BS = None
        continue

equity=pd.DataFrame({'profit':np.cumsum(profit_list),'profitfee':np.cumsum(profit_fee_list)},index=timeList)
print(equity)

equity.plot(grid=True, figsize=(12, 6))
plt.show()

