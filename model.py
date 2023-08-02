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

C=10
penalty='l2'
solver='liblinear'

print('\n Model :', 'Logistic Regression\n')

pipeline = Pipeline(steps = [('scale',StandardScaler()), ("LR",LogisticRegression(random_state=random_state, C=C, penalty=penalty, solver=solver))])
pipeline=pipeline.fit(X_train,y_train)
lr = LogisticRegression(random_state=random_state, C=C, penalty=penalty, solver= solver)
lr = lr.fit(X_res, y_res)
cross_val_scores = cross_val_score(lr, X_res, y_res, cv=5)
print("Cross-Validation Scores:", cross_val_scores)
print("Mean Cross-Validation Score:", np.mean(cross_val_scores))



# TEST ON UNSEEN X_test
print('\nTEST ON UNSEEN X_test\n')

prediction_label_X_test = pipeline.predict(X_test)

print(classification_report(y_test, prediction_label_X_test))
print('Accuracy Score: ', accuracy_score(y_test, prediction_label_X_test))
print('Recall Score: ', recall_score(y_test, prediction_label_X_test))
print('\nConfusion matrix: \n', confusion_matrix(y_test, prediction_label_X_test))





# TEST ON UNSEEN X_test (20 last)
print('\nTEST ON UNSEEN X_test (20 last)\n')

prediction_label_X_test = pipeline.predict(X_test.tail(20))

real  = y_test.tail(20).values.tolist()
real = ''.join(str(num) for num in real)
pred = ''.join(str(num) for num in prediction_label_X_test)
print('\nReal laabels VS Predicted labels\n')
print( real)
print( pred, '\n')


print(classification_report(y_test.tail(20), prediction_label_X_test))
print('Accuracy Score: ', accuracy_score(y_test.tail(20), prediction_label_X_test))
print('Recall Score: ', recall_score(y_test.tail(20), prediction_label_X_test))
print('\nConfusion matrix: \n', confusion_matrix(y_test.tail(20), prediction_label_X_test))



# TEST On UNSEEN EXTERNAL DATASET
print('\nTEST On UNSEEN EXTERNAL DATASE\n')

dt = pd.read_csv('data_test.csv')
X_test_dt, y_test_dt = transformation_pipeline_nores(dt)
prediction_label_dt = pipeline.predict(X_test_dt)
print('\nReal laabels VS Predicted labels\n')
print('1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0')
print(prediction_label_dt, '\n')


print(classification_report(y_test_dt, prediction_label_dt))
print('Accuracy Score: ', accuracy_score(y_test_dt, prediction_label_dt))
print('Recall Score: ', recall_score(y_test_dt, prediction_label_dt))
print('\nConfusion matrix: \n', confusion_matrix(y_test_dt, prediction_label_dt))




param_grid = {
    # 'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'C': [ 1, 10, 100],
    'penalty': [ 'l2'],
    'solver': ['liblinear', 'lbfgs']
}
logistic_regression = LogisticRegression(random_state=random_state)
grid_search = GridSearchCV(estimator=logistic_regression, param_grid=param_grid, scoring='roc_auc', cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)



pkl.dump(pipeline, open('model.pkl', 'wb'))


