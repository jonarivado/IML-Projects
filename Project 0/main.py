from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

trainset = pd.read_csv('train.csv')
X = trainset.drop(trainset.columns[[0, 1]], axis=1)
y = pd.DataFrame(trainset['y'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

regr = LinearRegression(fit_intercept=True)
regr.fit(X_train, y_train)

y_pred = pd.DataFrame(regr.predict(X_test))

RMSE = mean_squared_error(y_test, y_pred)**0.5
display(RMSE)

plt.scatter(y_test, y_pred)

X_sol = pd.read_csv('test.csv')
y_sol = pd.DataFrame()
y_sol['Id'] = X_sol['Id']
X_sol = X_sol.drop(labels='Id', axis=1)
y_sol['y'] = pd.DataFrame(regr.predict(X_sol))

y_sol.to_csv('submission.csv', index=False)
