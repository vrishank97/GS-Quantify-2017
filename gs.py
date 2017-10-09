import pandas as pd
import matplotlib as plt
import numpy as np
import xgboost as xgb

train = pd.read_csv("gcTrianingSet.csv")
sub= pd.read_csv("gcPredictionFile.csv")
sub.fillna(0, inplace=True)
X_train = train[['initialUsedMemory', 'initialFreeMemory', 'query token','cpuTimeTaken','gcRun']]
y_train1=train['finalUsedMemory']
y_train2=train['finalFreeMemory']
y_train3=train['gcRun']

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

X_train=pd.get_dummies(X_train)
sub=pd.get_dummies(sub)


from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, y_train1, test_size=0.33)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train, y_train2, test_size=0.33)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_train.drop(['gcRun'], axis=1), y_train3, test_size=0.33)
from sklearn.metrics import confusion_matrix

dtrain1 = xgb.DMatrix(X_train1, y_train1)
dtrain2 = xgb.DMatrix(X_train2, y_train2)
dtrain3 = xgb.DMatrix(X_train3, y_train3)
dtest1 = xgb.DMatrix(X_test1)
dtest2 = xgb.DMatrix(X_test2)
dtest3 = xgb.DMatrix(X_test3)

xgb_params1 = {
    'n_trees': 500, 
    'eta': 0.01,
    'max_depth': 4,
    'min_child_weight': 1,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
'''
cv_result = xgb.cv(xgb_params1, 
                   dtrain1,
		               nfold=5, 
                   num_boost_round=2000,
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False,
		               seed=0
                  )
'''
xgb_params2 = {
    'n_trees': 500, 
    'eta': 0.01,
    'max_depth': 4,
    'min_child_weight': 1,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
'''
cv_result = xgb.cv(xgb_params2, 
                   dtrain2,
		               nfold=5, 
                   num_boost_round=2000,
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False,
		               seed=0
                  )
'''
xgb_params3 = {
    'n_trees': 500, 
    'eta': 0.01,
    'max_depth': 4,
    'min_child_weight': 1,
    'subsample': 0.95,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'silent': 1
}
'''
cv_result = xgb.cv(xgb_params3, 
                   dtrain3,
		               nfold=5, 
                   num_boost_round=2000,
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False,
		               seed=0
                  )
'''
model1 = xgb.train(xgb_params1, dtrain1, num_boost_round=2050)
y_pred1 = model1.predict(dtest1)
from sklearn.metrics import r2_score
print r2_score(y_test1, y_pred1)

model2 = xgb.train(xgb_params2, dtrain2, num_boost_round=2050)
y_pred2 = model2.predict(dtest2)
from sklearn.metrics import r2_score
print r2_score(y_test2, y_pred2)

model3 = xgb.train(xgb_params3, dtrain3, num_boost_round=2050)
y_pred3 = model3.predict(dtest3)
'''
col1=model1.predict(xgb.DMatrix(sub.iloc[[0]]))
col2=model2.predict(xgb.DMatrix(sub.iloc[[0]]))
sub.set_value(1,'initialUsedMemory',col1)
sub.set_value(1,'initialFreeMemory',col2)
'''
for i, row in sub.iterrows():
	col1=model1.predict(xgb.DMatrix(sub.iloc[[i]]))
	col2=model2.predict(xgb.DMatrix(sub.iloc[[i]]))
	sub.set_value(i+1,'initialUsedMemory',col1)
	sub.set_value(i+1,'initialFreeMemory',col2)
	col3=model3.predict(xgb.DMatrix(sub.iloc[[i+1]].drop(['gcRun'], axis=1)))
	if col3>0.45:
		sub.set_value(i+1,'gcRun',1)
	if col3<0.45:
		sub.set_value(i+1,'gcRun',0)

sub[['initialFreeMemory','gcRun']].to_csv('./GSsubmission.csv', index_label=None)


