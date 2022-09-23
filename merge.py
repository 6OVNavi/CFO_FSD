import pandas as pd

y_train=pd.read_csv('y_train_fixed.csv')
train=pd.read_parquet('train_datesV.parquet')

#print(y_train)
#print(train)
print(len(train))
temp=pd.merge(y_train,train,how='left',left_on=['year','month'],right_on=['year','month'])
print(len(temp))
temp.to_parquet('train_merged.parquet')
