import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
train_data=pd.read_csv('train.csv')
y=pd.DataFrame(train_data['y'])
X=pd.DataFrame(train_data.drop(train_data.columns[[0,1]],axis=1))

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)

#fit model
reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_test,y_test))
print(reg.coef_)
#print(reg.intercept_)
#y_pred = reg.predict(X_test)

