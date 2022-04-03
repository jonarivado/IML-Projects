import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import mean_squared_error

#load dataset
train_df=pd.read_csv(r'C:\Users\Manue\Documents\ETH\Master\2.semester\introduction to machinelearning\code projects\1.a\train.csv')
y=pd.DataFrame(train_df['y'])
X=train_df.drop(labels='y',axis=1)

lambdas = [0.1, 1, 10, 100, 200]
avg_acc = np.zeros(5)

for i in range(5):

    kf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
    
    model = Ridge(alpha=[lambdas[i]], max_iter=10000, tol=0.00001)

    result = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv = kf)
    result = np.absolute(result)
    avg_acc[i] = np.mean(result)
    print("Average accuracy: {}".format(result.mean()))

RMSE = pd.DataFrame(avg_acc)
RMSE.to_csv('submission.csv', index=False, header= None)