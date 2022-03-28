import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#load dataset
train_data=pd.read_csv('train.csv')
y=pd.DataFrame(train_data['y'])
X=pd.DataFrame(train_data.drop(train_data.columns[[0,1]],axis=1))

#feature transformation
#quadratic x6-x10
X_squared = X**2
X_squared.columns=['x6', 'x7','x8', 'x9', 'x10']
X_trafo = pd.concat([X, X_squared],axis=1)

#exponential x11-x15
X_exp = np.exp(X)
X_exp.columns=['x11', 'x12','x13', 'x14', 'x15']
X_trafo = pd.concat([X_trafo, X_exp],axis=1)

#cosine x16-x20
X_cos = np.cos(X)
X_cos.columns=['x16', 'x17','x18', 'x19', 'x20']
X_trafo = pd.concat([X_trafo, X_cos],axis=1)

#constant x21
X_trafo['x21'] = np.ones_like(700)

#regression
X_train, X_test, y_train, y_test = train_test_split(
X_trafo, y, test_size=0.2, random_state=42)

#fit model
reg = LinearRegression().fit(X_train, y_train)
print(reg.score(X_test,y_test))
#print(pd.DataFrame(reg.coef_,axis=0))
weights_df=pd.DataFrame(data=np.transpose(reg.coef_))
weights_df.to_csv('submission.csv',index=False, header=False)