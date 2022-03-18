import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
# load the dataset
train_df=pd.read_csv(r'C:\Users\Manue\Documents\ETH\Master\2.semester\introduction to machinelearning\code projects\1.a\train.csv')
y=pd.DataFrame(train_df['y'])
X=train_df.drop(labels='y',axis=1)

lambdas=[0.1,1,10,100,200]
score_array=np.empty(5)
for i in range(5):
    model = Ridge(alpha=lambdas[i])
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    scores = np.absolute(scores)
    score_array[i] =np.mean(scores)
print(score_array)
score_df = pd.DataFrame(score_array)
score_df.to_csv('submission.csv', index=False, header= None)