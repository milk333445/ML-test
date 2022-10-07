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

print(data.iloc[0:8])


#抓到3分k

print(data.resample(rule='3T',closed='left',label='left').first().iloc[0:3])
#print(data.resample(rule='3T',closed='right',label='left').first().iloc[0:3])
#print(data.resample(rule='3T',closed='left',label='right').first().iloc[0:3])


# 16:03 的 high 是 16:03~16:05 的 high
print(data.resample(rule='3T',closed='left',label='left').max().iloc[0:3])
# 16:03 的 low 是 16:03~16:05 的 low
print(data.resample(rule='3T', closed='left', label='left').min().iloc[0:3])
# 16:03 的 close 要抓到 16:05 的 close
print(data.resample(rule='3T', closed='left', label='left').last().iloc[0:3])
# 16:03 的 volume 是 16:03~16:05 的 sum
print(data.resample(rule='3T', closed='left', label='left').sum().iloc[0:3])


#3分k資料
d1=data.resample(rule='3T',closed='left',label='left').first()[['open']]
d2=data.resample(rule='3T',closed='left',label='left').max()[['high']]
d3=data.resample(rule='3T', closed='left', label='left').min()[['low']]
d4=data.resample(rule='3T', closed='left', label='left').last()[['close']]
d5=data.resample(rule='3T', closed='left', label='left').sum()[['volume']]
df=pd.concat([d1,d2,d3,d4,d5],axis=1)
print(df)
#一了時k
rule = '1H'

d1 = data.resample(rule=rule, closed='left', label='left').first()[['open']]
d2 = data.resample(rule=rule, closed='left', label='left').max()[['high']]
d3 = data.resample(rule=rule, closed='left', label='left').min()[['low']]
d4 = data.resample(rule=rule, closed='left', label='left').last()[['close']]
d5 = data.resample(rule=rule, closed='left', label='left').sum()[['volume']]

df1 = pd.concat([d1,d2,d3,d4,d5], axis=1)
print(df1)

#一天
rule = '1D'

d1 = data.resample(rule=rule, closed='left', label='left').first()[['open']]
d2 = data.resample(rule=rule, closed='left', label='left').max()[['high']]
d3 = data.resample(rule=rule, closed='left', label='left').min()[['low']]
d4 = data.resample(rule=rule, closed='left', label='left').last()[['close']]
d5 = data.resample(rule=rule, closed='left', label='left').sum()[['volume']]

df2 = pd.concat([d1,d2,d3,d4,d5], axis=1)
print(df2)
