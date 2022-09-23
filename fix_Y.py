import pandas as pd

Y=pd.read_excel('Y_train.xlsx')

print(Y)
new_d=[]
for vals in range(1,len(Y.columns)):
    new_d.append([int(str(Y.iloc[0][vals])[:4]),int(str(Y.iloc[0][vals])[5:7]),float(Y.iloc[1][vals])])

y_train=pd.DataFrame(new_d,columns=['year','month','target'])

print(y_train)

y_train.to_csv('y_train_fixed.csv',index=False)
