from sklearn.linear_model import RidgeCV
import pandas as pd


#load dataset
train_df=pd.read_csv(r'C:\Users\Manue\Documents\ETH\Master\2.semester\introduction to machinelearning\code projects\1.a\train.csv')
y=pd.DataFrame(train_df['y'])
X=train_df.drop(labels='y',axis=1)

#ridge regression
clf = RidgeCV(alphas=[0.1] , cv=10).fit(X, y)
print(clf.score(X, y))
