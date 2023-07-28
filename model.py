import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.fixes import loguniform
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import (accuracy_score,
                             auc,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             confusion_matrix)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.classifier import StackingCVClassifier
from sklearn.svm import SVC
from transformation_pipeline import transformation_pipeline
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.svm import SVC, LinearSVC
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from transformation_pipeline import transformation_pipeline,transformation_pipeline_nores


random_state = 42

df = pd.read_csv("data_strokes_prediction.csv")

X_res,y_res = transformation_pipeline(df)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.33, random_state=random_state)





# XGB_BEST_PARAMS = {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 180}
# pipeline = Pipeline(steps = [('scale',StandardScaler()), ("XGBC",XGBClassifier(random_state=random_state, params=XGB_BEST_PARAMS))])
# pipeline=pipeline.fit(X_train,y_train)
#
#
# print('X_train\n', X_train)
# print('y_train\n', y_train)
#
# xg = XGBClassifier(random_state=random_state, params=XGB_BEST_PARAMS)
# xg = xg.fit(X_res, y_res)
# cross_val_scores = cross_val_score(xg, X_res, y_res, cv=5)
# print("Cross-Validation Scores:", cross_val_scores)
# print("Mean Cross-Validation Score:", np.mean(cross_val_scores))
# # Hyperparameter tuning using GridSearch
# estimator = XGBClassifier(learning_rate= 0.1, max_depth=9, n_estimators= 180)
# parameters = {
#     'max_depth': range(2, 10, 1),
#     'n_estimators': range(60, 220, 40),
#     'learning_rate': [0.1, 0.01, 0.05] }
# grid_search = GridSearchCV(estimator=estimator, param_grid=parameters, scoring='roc_auc', n_jobs=10, cv=10, verbose=True)
# grid_search.fit(X_train, y_train)
# # Get the best hyperparameters from GridSearch
# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)
#
# tuned_pred = pipeline.predict(X_test)
#
#
# dt = pd.read_csv('data_test.csv')
# print('dt\n', dt)
#
# X_test_dt, y_test_dt = transformation_pipeline_nores(dt)
#
#
# prediction_label_dt = pipeline.predict(X_test_dt)
# print('prediction_label_dt', prediction_label_dt)
#
#
# print(classification_report(y_test_dt, prediction_label_dt))
# print('Accuracy Score: ', accuracy_score(y_test_dt, prediction_label_dt))
# print('Recall Score: ', recall_score(y_test_dt, prediction_label_dt))
# print('\nConfusion matrix: \n', confusion_matrix(y_test_dt, prediction_label_dt))
# print('\n Model :', 'XGB')








# LGBM_BEST_PARAMS = { }
# XGB_BEST_PARAMS = {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 180}
# pipeline = Pipeline(steps = [('scale',StandardScaler()), ("XGBC",LGBMClassifier(random_state=random_state))])
# pipeline=pipeline.fit(X_train,y_train)
#
#
# print('X_train\n', X_train)
# print('y_train\n', y_train)
#
# xg = LGBMClassifier(random_state=random_state)
# xg = xg.fit(X_res, y_res)
# cross_val_scores = cross_val_score(xg, X_res, y_res, cv=5)
# print("Cross-Validation Scores:", cross_val_scores)
# print("Mean Cross-Validation Score:", np.mean(cross_val_scores))
#
#
# tuned_pred = pipeline.predict(X_test)
#
#
# dt = pd.read_csv('data_test.csv')
# print('dt\n', dt)
#
# X_test_dt, y_test_dt = transformation_pipeline_nores(dt)
#
#
# prediction_label_dt = pipeline.predict(X_test_dt)
# print('prediction_label_dt', prediction_label_dt)
#
#
# print(classification_report(y_test_dt, prediction_label_dt))
# print('Accuracy Score: ', accuracy_score(y_test_dt, prediction_label_dt))
# print('Recall Score: ', recall_score(y_test_dt, prediction_label_dt))
# print('\nConfusion matrix: \n', confusion_matrix(y_test_dt, prediction_label_dt))
# print('\n Model :', 'XGB')



LR_BEST_PARAMS = { }
pipeline = Pipeline(steps = [('scale',StandardScaler()), ("XGBC",LogisticRegression(random_state=random_state))])
pipeline=pipeline.fit(X_train,y_train)


print('X_train\n', X_train)
print('y_train\n', y_train)

lr = LogisticRegression(random_state=random_state, C= 10, penalty= 'l1', solver= 'liblinear')
lr = lr.fit(X_res, y_res)
cross_val_scores = cross_val_score(lr, X_res, y_res, cv=5)
print("Cross-Validation Scores:", cross_val_scores)
print("Mean Cross-Validation Score:", np.mean(cross_val_scores))


tuned_pred = pipeline.predict(X_test)


dt = pd.read_csv('data_test.csv')
print('dt\n', dt)

X_test_dt, y_test_dt = transformation_pipeline_nores(dt)


prediction_label_dt = pipeline.predict(X_test_dt)
print('prediction_label_dt', prediction_label_dt)


print(classification_report(y_test_dt, prediction_label_dt))
print('Accuracy Score: ', accuracy_score(y_test_dt, prediction_label_dt))
print('Recall Score: ', recall_score(y_test_dt, prediction_label_dt))
print('\nConfusion matrix: \n', confusion_matrix(y_test_dt, prediction_label_dt))
print('\n Model :', 'LR')



param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs']
}
logistic_regression = LogisticRegression(random_state=random_state, params={'C': 10, 'penalty': 'l1', 'solver': 'liblinear'})
grid_search = GridSearchCV(estimator=logistic_regression, param_grid=param_grid, scoring='roc_auc', cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)






