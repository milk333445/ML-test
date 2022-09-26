import yfinance as yf
import pandas as pd
import pandas_datareader as data
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#台積電
target_stock='2330.TW'
start_date=datetime(2021,7,23)
end_date=datetime(2022,7,23)
dfy=data.get_data_yahoo([target_stock],start_date,end_date)
dfy.to_csv('台積電資料.csv')
df1=pd.read_csv('台積電資料.csv')
df=pd.read_csv('ML資料集(調參).csv', parse_dates=True)
df['Date'] = pd.to_datetime(df['Date'])
print(df)
df=df.rename(columns={'0':'prediction'})
condition=df['2330.TW']>df['2330.TW'].shift(1)
df['label']=condition
df['label']=df['label'].astype(int)
print(df)
list=[]
AP=1
month_r=int(7)
percentage_r=[]
for i in range(1,245):
    if df['prediction'][i]==1:
        AP=AP*(1+df['fluctuation'][i])*0.997
    else:
        AP=AP*(1-df['fluctuation'][i])*0.997
    if month_r == 13:
        month_r=1
    if df['Date'][i].month == month_r:
        percentage_r.append((AP) - 1)
        month_r += 1
    print(AP-1)
    list.append(AP-1)
print(percentage_r)
print(list)



line = plt.plot(list, color='#AE81FF', linestyle='solid', label='ML model profit')
plt.xlabel('day')
plt.ylabel('accumulated profit')
plt.legend(loc='upper center')
plt.title('ML model strategy')
plt.show()