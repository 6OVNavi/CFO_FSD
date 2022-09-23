# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np

import random,os

seed=42

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(seed)

pd.set_option('display.max_columns', None)

train=pd.read_parquet('train_merged.parquet')

train=train.drop('DateObserve',axis=1)

print(train.isna().sum())

print(list(train.columns))

print(train)

train['CurrentPrice']=train['CurrentPrice'].fillna(1e9)

train.loc[train.StockStatus=='InStock', 'StockStatus']=1
train.loc[train.StockStatus=='OutOfStock', 'StockStatus']=0

#train=pd.get_dummies(train,columns=['StockStatus'])

from catboost import CatBoostRegressor

X=train.drop('target',axis=1)
y=train['target']



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)


cb=CatBoostRegressor(iterations=60,custom_metric=['RMSE','R2'],random_state=seed)
cb.fit(X_train,y_train,eval_set=(X_val,y_val))

print(cb.best_score_,cb.best_iteration_)
