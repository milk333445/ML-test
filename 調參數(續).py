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

#台積電
target_stock='2330.TW'
start_date=datetime(2019,1,2)
end_date=datetime(2022,7,23)
dfy=data.get_data_yahoo([target_stock],start_date,end_date)
#print(dfy)
#print(dfy['Close'])

#中砂
target_stock='1560.TW'
dfx1=data.get_data_yahoo([target_stock],start_date,end_date)
#勝一
target_stock='1773.TW'
dfx2=data.get_data_yahoo([target_stock],start_date,end_date)
#光洋科
target_stock='1785.TWO'
dfx3=data.get_data_yahoo([target_stock],start_date,end_date)
#漢唐
target_stock='2404.TW'
dfx4=data.get_data_yahoo([target_stock],start_date,end_date)
#盟立
target_stock='2464.TW'
dfx5=data.get_data_yahoo([target_stock],start_date,end_date)
df=pd.concat([dfy['Close'],dfx1['Close'],dfx2['Close'],dfx3['Close'],dfx4['Close'],dfx5['Close']],axis=1)
#print(df)

condition=df['2330.TW']>df['2330.TW'].shift(1)
#print(condition)
df['y']=condition
#print(df)
df['y']=df['y'].astype(int)
#print(df)
#print(df.isnull().sum())

#切資料
x=df.drop(['y','2330.TW'],axis=1)
y=df['y']

def split_data(x,y,test_size):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size)
    return np.array(x_train),np.array(x_test),np.array(y_train),np.array(y_test)

x_train,x_test,y_train,y_test=split_data(x,y,test_size=0.3)
x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

#最終結果
classifier=RandomForestClassifier(n_estimators=91,
                                 criterion='entropy',
                                 min_samples_split=2,
                                 max_depth=50,
                                 random_state=0,
                                 min_samples_leaf=2,
                                 max_features='auto',
                                 bootstrap='False'
                                 )

classifier=classifier.fit(x_train,y_train)

#特徵值
import matplotlib.pyplot as plt
feature_names=x.keys().tolist()
result=pd.DataFrame(
    {'feature':feature_names,
    'feature_importance':classifier.feature_importances_.tolist()
    }
)
result=result.sort_values(by=['feature_importance'],ascending=False)
print(result)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.02*height, '%f'%float(height),
                 ha='center', va='bottom')
plt.style.use('ggplot')
fig = plt.figure(figsize=(100, 6))
gini = plt.bar(result.index, result['feature_importance'], align='center')
plt.xlabel('Feature')
plt.ylabel('Feature Importance')
plt.xticks(result.index, result['feature'])
autolabel(gini)
plt.show()

test_prediction=classifier.predict(x_test)
print(test_prediction)
import seaborn as sns
confusion_matrix_result= confusion_matrix(y_test, test_prediction)
print(confusion_matrix_result)
sns.set()
f,ax=plt.subplots()
sns.heatmap(confusion_matrix_result,annot=True,ax=ax)

ax.set_title('confusion matrix')
ax.set_xlabel('predict')
ax.set_ylabel('true')
plt.show()

from sklearn.metrics import f1_score
print(f1_score(y_test,test_prediction,average='macro'))




#交易策略
start_date=datetime(2021,7,23)
end_date=datetime(2022,7,23)
#中砂
target_stock='1560.TW'
dfx1=data.get_data_yahoo([target_stock],start_date,end_date)
#勝一
target_stock='1773.TW'
dfx2=data.get_data_yahoo([target_stock],start_date,end_date)
#光洋科
target_stock='1785.TWO'
dfx3=data.get_data_yahoo([target_stock],start_date,end_date)
#漢唐
target_stock='2404.TW'
dfx4=data.get_data_yahoo([target_stock],start_date,end_date)
#盟立
target_stock='2464.TW'
dfx5=data.get_data_yahoo([target_stock],start_date,end_date)
df_trade_x_test=pd.concat([dfx1['Close'],dfx2['Close'],dfx3['Close'],dfx4['Close'],dfx5['Close']],axis=1)
print(df_trade_x_test)
df_trade_x_test_prediction=classifier.predict(df_trade_x_test)
print(df_trade_x_test_prediction)
df_trade_x_test_prediction=pd.DataFrame(df_trade_x_test_prediction,index=df_trade_x_test.index)
print(df_trade_x_test_prediction)
#台積電
target_stock='2330.TW'
start_date=datetime(2021,7,23)
end_date=datetime(2022,7,23)
dfy=data.get_data_yahoo([target_stock],start_date,end_date)
print(dfy['Close'])

df_trade_data=pd.concat([dfy['Close'],df_trade_x_test_prediction],axis=1)
print(df_trade_data)

df_trade_data['fluctuation']=(df_trade_data['2330.TW']/df_trade_data['2330.TW'].shift(1))-1
print(df_trade_data)
df_trade_data=df_trade_data.fillna(0)
print(df_trade_data)
df_trade_data.to_csv('ML資料集(調參).csv')