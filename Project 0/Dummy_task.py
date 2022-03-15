import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#load dataset
data=pd.read_csv(r'C:\Users\Manue\Documents\GitHub\IML-Projects\Project 0\train.csv')
y=data['y']
train=data.iloc[0:10000,2:12]
X=np.asarray(train)
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)

#fit model
reg = LinearRegression().fit(X_train, y_train)
#print(reg.score(X_train,y_train))
#print(reg.coef_)
#print(reg.intercept_)
y_pred = reg.predict(X_test)
#RMSE=np.sqrt(((y_pred - y_test) ** 2).mean())

#load test data
test_data=pd.read_csv(r'C:\Users\Manue\Documents\GitHub\IML-Projects\Project 0\test.csv')
submissiondf = pd.DataFrame()
submissiondf['Id']= test_data['Id']
test=test_data.iloc[0:2000,1:11]
test=np.asarray(test)

#make predictions
y_predict=pd.DataFrame(reg.predict(test))
submissiondf['y']=y_predict

#store predictions in a .csv file
submissiondf.to_csv(r'C:\Users\Manue\Documents\GitHub\IML-Projects\Project 0\submission.csv', index=False)
