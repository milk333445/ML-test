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
fund=100
money=100
feeRate=0.003
length=30
stdTimes=0.5


rule='1D'
d1 = df_outSample.resample(rule=rule, closed='left', label='left').first()[['open']]
d2 = df_outSample.resample(rule=rule, closed='left', label='left').max()[['high']]
d3 = df_outSample.resample(rule=rule, closed='left', label='left').min()[['low']]
d4 = df_outSample.resample(rule=rule, closed='left', label='left').last()[['close']]
d5 = df_outSample.resample(rule=rule, closed='left', label='left').sum()[['volume']]

df= pd.concat([d1,d2,d3,d4,d5], axis=1)
#print(df)

df['ma']=df['close'].rolling(window=length,center=False).mean()
df['std'] = df['close'].rolling(window=length, center=False).std()

BS = None
buy = []
sell = []
sellshort = []
buytocover = []
profit_list = [0]
profit_fee_list = [0]
profit_fee_list_realized = []

rule='1H'
df_hour=data.resample(rule=rule,closed='left',label='left').first()[['open']]
df_funding=pd.concat([df_hour,funding],axis=1)
df_funding=df_funding.fillna(method='bfill',limit=2).fillna(0)
def fundingPayment(df_funding,side,unit,t1,t2):
    if len(df_funding.loc[t1:t2])==0:
        fee=0
    else:
        fr=np.array(df_funding.loc[t1:t2])
        fee=unit*np.dot(fr[:,0],fr[:,1])
    if side=='long':
        return -fee
    elif side=='short':
        return fee
for i in range(len(df)):
    if i==len(df)-1:
        break
    if BS==None:
        profit_list.append(0)
        profit_fee_list.append(0)
        if df['close'][i] > df['ma'][i]+stdTimes*df['std'][i]:
            tempSize=money/df['open'][i+1]
            BS='B'
            t=i+1
            buy.append(t)
            t1=df.index[i+1]
        elif df['close'][i] < df['ma'][i]-stdTimes*df['std'][i]:
            tempSize=money/df['open'][i+1]
            BS='S'
            t=i+1
            sellshort.append(t)
            t1=df.index[i+1]

    elif BS=='B':
        profit=tempSize*(df['open'][i+1]-df['open'][i])
        profit_list.append(profit)
        t2=df.index[i+1]
        fundingFee=fundingPayment(df_funding,'long',tempSize,df.index[t],t2)

        if df['close'][i] <= df['ma'][i] or i==len(df)-2:
            pl_round=tempSize*(df['open'][i+1]-df['open'][t])
            profit_fee=profit-money*feeRate-(money+pl_round)*feeRate+fundingFee
            profit_fee_list.append(profit_fee)
            sell.append(i+1)
            BS=None

            profit_fee_realized = pl_round - money * feeRate - (money + pl_round) * feeRate + fundingFee
            profit_fee_list_realized.append(profit_fee_realized)

        else:
            profit_fee=profit
            profit_fee_list.append(profit)
            t1=df.index[i+1]

    elif BS=='S':
        profit=tempSize*(df['open'][i]-df['open'][i+1])
        profit_list.append(profit)
        t2=df.index[i+1]
        fundingFee=fundingPayment(df_funding,'short',tempSize,df.index[t],t2)

        if df['close'][i] >= df['ma'][i] or i==len(df)-2:
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
equity = pd.DataFrame({'profit':np.cumsum(profit_list), 'profitfee':np.cumsum(profit_fee_list)}, index=df.index)
print(equity)
equity.plot(grid=True, figsize=(12, 6))
plt.show()

equity['equity']=equity['profitfee']+fund
equity['drawdown_percent']=(equity['equity']/equity['equity'].cummax())-1
equity['drawdown'] = equity['equity'] - equity['equity'].cummax()

profit=equity['profitfee'].iloc[-1]
ret=equity['equity'][-1]/equity['equity'][0]-1
mdd=abs(equity['drawdown_percent'].min())
calmarRatio = ret / mdd
tradeTimes=len(buy)+len(sellshort)
winRate=len([i for i in profit_fee_list_realized if i > 0])/len([i for i in profit_fee_list_realized if i > 0])
profitFactor=sum([i for i in profit_fee_list_realized if i>0])/sum([i for i in profit_fee_list_realized if i<0])
winLossRatio = np.mean([i for i in profit_fee_list_realized if i > 0]) / abs(np.mean([i for i in profit_fee_list_realized if i < 0]))
print(f'profit:${np.round(profit,2)}')
print(f'return: {np.round(ret,4)*100}%')
print(f'mdd: {np.round(mdd,4)*100}%')
print(f'calmarRatio: {np.round(calmarRatio,2)}')
print(f'tradeTimes: {tradeTimes}')
print(f'winRate: {np.round(winRate,4)*100}%')
print(f'profitFactor: {np.round(profitFactor,2)}')
print(f'winLossRatio: {np.round(winLossRatio,2)}')


fig,ax=plt.subplots(figsize=(16,6))
equity['profitfee'].plot(label='Total Profit',ax=ax,c='c',grid=True)
plt.fill_between(equity['drawdown'].index,equity['drawdown'],0,facecolor='r',label='Drawdown', alpha=0.5)
high_index = equity[equity['profitfee'].cummax() == equity['profitfee']].index
plt.scatter(high_index,equity['profitfee'].loc[high_index],c='#02ff0f', label = 'High')
plt.legend()
plt.ylabel('Accumulated Profit')
plt.xlabel('Time')
plt.title('Profit & Drawdown',fontsize  = 16)
plt.show()

fig,ax=plt.subplots(figsize=(16,6))
df['close'].plot(label='Close Price',ax=ax,c='gray',grid=True,alpha=0.8)
plt.scatter(df['close'].iloc[buy].index,df['close'].iloc[buy],c = 'orangered', label = 'Buy', marker='^', s=60)
plt.scatter(df['close'].iloc[sell].index, df['close'].iloc[sell],c = 'orangered', label = 'Sell', marker='v', s=60)
plt.scatter(df['close'].iloc[sellshort].index, df['close'].iloc[sellshort],c = 'limegreen', label = 'Sellshort', marker='v', s=60)
plt.scatter(df['close'].iloc[buytocover].index, df['close'].iloc[buytocover],c = 'limegreen', label = 'Buytocover', marker='^', s=60)

plt.legend()
plt.ylabel('USD')
plt.xlabel('Time')
plt.title('Price Movement',fontsize  = 16)
plt.show()

#iloc跟loc的差別
#iloc是根據行號找
#loc是根據標籤找
print(equity['profitfee'].loc[high_index])
print(df['close'].iloc[buy].index)
print(high_index)
print(buy)