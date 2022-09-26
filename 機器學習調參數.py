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

#隨機森林

n_estimators=[int(x) for x in np.linspace(60,160,num=20)]
max_features=['auto','sqrt']
max_depth=[int(x) for x in np.linspace(10,100,num=10)]
max_depth.append(None)
min_samples_split = [2, 4, 5, 7, 8, 10]
min_samples_leaf = [1, 2, 3, 4, 5, 6]
bootstrap = [True, False]
criterion = ['entropy']
random_state = [0]

random_grid={
    'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf,
    'bootstrap':bootstrap,
    'criterion':criterion,
    'random_state':random_state
}

rfc=RandomForestClassifier()
rfc_random_search=RandomizedSearchCV(estimator=rfc,
                                     param_distributions=random_grid,
                                     n_iter=100,
                                     scoring='accuracy',
                                     cv=3,
                                     verbose=2,
                                     random_state=35,
                                     n_jobs=-1
                                     )
rfc_random_search.fit(x_train,y_train)
rfc_random_model=rfc_random_search.best_estimator_
print("Score of train set: % .10f" % (rfc_random_model.score(x_train, y_train)))
print("Score of test set: % .10f" % (rfc_random_model.score(x_test, y_test)))
print("Best score:{}".format(rfc_random_search.best_score_))
print("Best parameters:{}".format(rfc_random_search.best_params_))

