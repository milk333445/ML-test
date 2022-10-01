import yfinance as yf
import pandas as pd
import pandas_datareader as data
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble,metrics
from sklearn.metrics import classification_report,confusion_matrix
from xgboost.sklearn import XGBClassifier
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
yf.pdr_override()

#取得股票

#台達電
target_stock='2308.TW'
start_date=datetime(2011,9,23)
end_date=datetime(2021,9,23)
dfy=data.get_data_yahoo([target_stock],start_date,end_date)
#print(dfy)
#print(dfy['Close'])

#康舒
target_stock='6282.TW'
dfx1=data.get_data_yahoo([target_stock],start_date,end_date)
#致茂
target_stock='2360.TW'
dfx2=data.get_data_yahoo([target_stock],start_date,end_date)
#貿聯KY
target_stock='3665.TW'
dfx3=data.get_data_yahoo([target_stock],start_date,end_date)
#和大
target_stock='1536.TW'
dfx4=data.get_data_yahoo([target_stock],start_date,end_date)
#市電
target_stock='1503.TW'
dfx5=data.get_data_yahoo([target_stock],start_date,end_date)
#美琪瑪
target_stock='4721.TWO'
dfx6=data.get_data_yahoo([target_stock],start_date,end_date)
#大同
target_stock='2371.TW'
dfx7=data.get_data_yahoo([target_stock],start_date,end_date)
#廣龍
target_stock='1537.TW'
dfx8=data.get_data_yahoo([target_stock],start_date,end_date)
#中碳
target_stock='1723.TW'
dfx9=data.get_data_yahoo([target_stock],start_date,end_date)
#明基材
target_stock='8215.TW'
dfx10=data.get_data_yahoo([target_stock],start_date,end_date)
#聚合
target_stock='6509.TWO'
dfx11=data.get_data_yahoo([target_stock],start_date,end_date)
#精星
target_stock='8183.TWO'
dfx12=data.get_data_yahoo([target_stock],start_date,end_date)
#車王電
target_stock='1533.TW'
dfx13=data.get_data_yahoo([target_stock],start_date,end_date)

df=pd.concat([dfy['Close'],dfx1['Close'],dfx2['Close'],dfx3['Close'],dfx4['Close'],dfx5['Close'],dfx6['Close'],dfx7['Close'],dfx8['Close'],dfx9['Close'],dfx10['Close'],dfx11['Close'],dfx12['Close'],dfx13['Close']],axis=1)


df['2308.TW_shift']=df['2308.TW'].shift(1)
df=df.drop(index=['2011-09-23'])
condition=df['2308.TW_shift']>df['2308.TW_shift'].shift(1)
df['y']=condition
df['y']=df['y'].astype(int)
df['y']=df['y'].shift(-1)
df=df.drop(index=['2021-09-23'])
df['y']=df['y'].astype(int)
print(df)

df.to_csv('機器學習資料集(電動車).csv')
