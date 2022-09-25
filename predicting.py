import pandas as pd
import numpy as np

csv='tests/DS_test6(2020-06--2022-08-01).csv'

train=pd.read_csv(csv,sep='\t',skiprows=range(1, 64720336))
print(train)
def get_dt(x):
    date, time = x.split(' ')
    year,month,day = date.split('-')
    return f"{year}.{month}.{day} {time}"

train['DateObserve']=train['DateObserve'].apply(get_dt)

#число дней до текущего года
pref_days = [0]
for year in range(2020, 2022):
    pref_days.append(pd.Timestamp(f'{year}-12-31').dayofyear)
print(pref_days)
for i in range(1, len(pref_days)):
    pref_days[i] += pref_days[i-1]
print(pref_days)

#признаки из datetime
def prepare(x):
    x = x.split(' ')
    date = x[0].split('.')
    time = x[1].split(':')
    year,month,day = map(int, date)
    timestamp = pd.Timestamp(f'{year}-{month}-{day}')
    hour = int(time[0])
    dayofyear = timestamp.dayofyear
    dayofweek = timestamp.dayofweek
    ind_date = dayofyear + pref_days[year - 2020] - 1
    ind_hour = ind_date * 24 + hour
    if dayofweek >= 5:
        days_to_weekend = 0
    else:
        days_to_weekend = min(dayofweek + 1, abs(5 - dayofweek))
    return year, month, day, hour, dayofyear, dayofweek, dayofweek>=5, (month-1) // 3, ind_date, ind_hour, days_to_weekend

add_columns = ['year', 'month', 'day', 'hour', 'dayofyear', 'dayofweek', 'is_weekend', \
               'season', 'ind_date', 'ind_hour', 'days_to_weekend']

train[add_columns] = list(train['DateObserve'].apply(prepare))
for col in add_columns:
    train[col] = train[col].astype(np.int32)

print('1')

train['CurrentPrice']=train['CurrentPrice'].fillna(1e6)

train.loc[train.StockStatus=='InStock', 'StockStatus']=1
train.loc[train.StockStatus=='Instock', 'StockStatus']=1
train.loc[train.StockStatus=='OutOfStock', 'StockStatus']=0

train['CurrentPrice_cumsum']=train.groupby(by=['WebPriceId','month','year'])['CurrentPrice'].transform('cumsum')
train['CurrentPrice_mean']=train.groupby(by=['WebPriceId','month','year'])['CurrentPrice'].transform('mean')
train['CurrentPrice_std']=train.groupby(by=['WebPriceId','month','year'])['CurrentPrice'].transform('std')
train['CurrentPrice_median']=train.groupby(by=['WebPriceId','month','year'])['CurrentPrice'].transform('median')
train['CurrentPrice_var']=train.groupby(by=['WebPriceId','month','year'])['CurrentPrice'].transform('var')

print('2')

train = train.sort_values(by='ind_hour')
print(train)
train=train.drop('DateObserve',axis=1)

print('3')

import pickle
from catboost import CatBoostRegressor

with open('model_old.pkl','rb') as f:
    modelold=pickle.load(f)

pred=modelold.predict(train.iloc[-1])
print('4')
with open('model_adaptive.pkl','rb') as f:
    modelnew=pickle.load(f)

pred2=modelnew.predict(train.iloc[-1])

print(pred,pred2)
