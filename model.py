import pandas as pd
import numpy as np
import pickle as pkl
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pycaret.classification import *
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import (accuracy_score,
                             auc,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             confusion_matrix)

random_state = 42

#load data


df = pd.read_csv("data_strokes_prediction.csv")

#drop gender other
df = df[(df['gender'] != 'Other')]

# drop outliers in age : 69768 (stroke at 1.32), 49669 (stroke at 14)
# outlier1 = df.loc[df['id'] == 69768]
# outlier2 = df.loc[df['id'] == 49669]
# print('outlier1',outlier1)
# print('outlier2',outlier2)
df=df.drop(df[df['id'] == 69768].index)
df=df.drop(df[df['id'] == 49669].index)

# fill null values with regression on numerical values
DT_bmi_pipe = Pipeline( steps=[
                               ('scale',StandardScaler()),
                               ('lr',DecisionTreeRegressor(random_state=random_state))
                              ])
X = df[['age','gender','bmi']].copy()
X.gender = X.gender.replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)
Missing = X[X.bmi.isna()]
X = X[~X.bmi.isna()]
Y = X.pop('bmi')
DT_bmi_pipe.fit(X,Y)
predicted_bmi = pd.Series(DT_bmi_pipe.predict(Missing[['age','gender']]),index=Missing.index)
df.loc[Missing.index,'bmi'] = round(predicted_bmi)

# transform : encode types object into categories
le = LabelEncoder()
df = df.drop(['id'], axis=1)  # drop useless column
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
       le.fit(df[col])
       df[col] = le.transform(df[col])

# define X as features, y as labels to predict
X,y = df.drop('stroke', axis = 1), df['stroke']


# undersampling + oversampling
over = SMOTE(sampling_strategy = 1)
under = RandomUnderSampler(sampling_strategy = 0.1)
print(X.shape,y.shape)
X_res_u, y_res_u = under.fit_resample(X, y)
print(X_res_u.shape,y_res_u.shape)
X_res, y_res = over.fit_resample(X_res_u, y_res_u)
print(X_res.shape,y_res.shape)




#split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=random_state)

# train
model,name = XGBClassifier(),'XGBC'
XGB_BEST_PARAMS = {'colsample_bytree': 0.6966652866539669,
                   'gamma': 5.1222111958093395,
                   'learning_rate': 0.28,
                   'max_depth': 16.0,
                   'min_child_weight': 1.0,
                   'nthread': 3.0, 'reg_alpha': 61.0,
                   'reg_lambda': 0.5855571439718152,
                   'scale_pos_weight': 4.0, 'subsample': 0.9}
pipeline = Pipeline(steps = [('scale',StandardScaler()), ("XGBC",XGBClassifier(random_state=random_state, params=XGB_BEST_PARAMS))])
pipeline.fit(X_train,y_train)


tuned_pred = pipeline.predict(X_test)
print(classification_report(y_test,tuned_pred))
print('Accuracy Score: ',accuracy_score(y_test,tuned_pred))
print('\nConfusion matrix: \n',confusion_matrix(y_test, tuned_pred))
print('\n Model :',name)


pkl.dump(pipeline, open('best_pipeline1.pkl', 'wb'))