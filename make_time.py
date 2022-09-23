import pandas as pd
import numpy as np

train=pd.read_parquet('train_p.parquet')
Y=pd.read_excel('Y_train.xlsx')
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
print(train['DateObserve'])
train[add_columns] = list(train['DateObserve'].apply(prepare))
for col in add_columns:
    train[col] = train[col].astype(np.int32)

print(list(Y.columns))
print(Y)

print(train)

train.to_parquet('train_datesV.parquet',index=False)
