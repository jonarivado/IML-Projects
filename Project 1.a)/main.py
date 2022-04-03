import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge, Lasso
# load the dataset
train_df=pd.read_csv('train.csv')
y=pd.DataFrame(train_df['y'])
X=train_df.drop(labels='y',axis=1)

lambdas = [0.1, 1, 10, 100, 200]
avg_acc = np.zeros(5)

for i in range(5):

    kf = RepeatedKFold(n_splits=10, n_repeats=10,random_state=42)
    
    model = Ridge(alpha=[lambdas[i]], max_iter=10000, tol=0.00001)

    result = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv = kf)
    result = np.absolute(result)
    avg_acc[i] = np.mean(result)
    print("Average accuracy: {}".format(result.mean()))

score_df = pd.DataFrame(avg_acc)
score_df.to_csv('submission.csv', index=False, header= None)