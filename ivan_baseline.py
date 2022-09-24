# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np

import random,os

import matplotlib.pyplot as plt
import seaborn as sns

seed=42

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(seed)

pd.set_option('display.max_columns', None)

train=pd.read_parquet('train_merged.parquet')

#print(dict(train['WebPriceId'].value_counts()).values())
#print(train['WebPriceId'].value_counts())
#train['WebPriceId'].value_counts().to_excel('val_counts.xlsx')
#exit(0)
#temp=train[(train['year']==2022)]
#print(train)

#print(temp)
#exit(0)
train=train.drop('DateObserve',axis=1)

print(train.isna().sum())

print(list(train.columns))

print(train)

train['CurrentPrice']=train['CurrentPrice'].fillna(0)

train.loc[train.StockStatus=='InStock', 'StockStatus']=1
train.loc[train.StockStatus=='Instock', 'StockStatus']=1
train.loc[train.StockStatus=='OutOfStock', 'StockStatus']=0

train['CurrentPrice_cumsum']=train.groupby(by=['WebPriceId','month','year'])['CurrentPrice'].transform('cumsum')
train['CurrentPrice_mean']=train.groupby(by=['WebPriceId','month','year'])['CurrentPrice'].transform('mean')
train['CurrentPrice_std']=train.groupby(by=['WebPriceId','month','year'])['CurrentPrice'].transform('std')
train['CurrentPrice_median']=train.groupby(by=['WebPriceId','month','year'])['CurrentPrice'].transform('median')
train['CurrentPrice_var']=train.groupby(by=['WebPriceId','month','year'])['CurrentPrice'].transform('var')

from sklearn.cluster import KMeans
import pickle

kmeans=KMeans(n_clusters=10, random_state=seed)

kmeans.fit(np.array(train['CurrentPrice']).reshape(-1,1))

with open('kmeans.pkl','wb') as f:
    pickle.dump(kmeans,f)

train['clustered_price']=kmeans.predict(np.array(train['CurrentPrice']).reshape(-1,1))

#train.to_parquet('in_between.parquet')

'''sns.set_theme(font_scale=0.6)
plt.figure(figsize=(25, 25))
sns.heatmap(train.corr(), annot=True,linewidths=1)
plt.tight_layout()
plt.show()'''

#train=pd.get_dummies(train,columns=['StockStatus'])

from catboost import CatBoostRegressor

train.to_parquet('train_end.parquet')

X=train.drop('target',axis=1)
y=train['target']

X_train=X[:64720336]
y_train=y[:64720336]

X_val=X[64720336:]#56386118 - 2022
y_val=y[64720336:]
print(X_train)
print(X_val)
from sklearn.model_selection import train_test_split

#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)


cb=CatBoostRegressor(iterations=25,custom_metric=['RMSE','R2','MAE'],random_state=seed,learning_rate=0.1)
cb.fit(X_train,y_train,eval_set=(X_val,y_val))

pred=cb.predict(X_val)

from sklearn.metrics import mean_squared_error

print(mean_squared_error([1.56],[np.mean(pred)]))

print(mean_squared_error([1.56],[pred[-1]]))

print(cb.best_score_,cb.best_iteration_)
