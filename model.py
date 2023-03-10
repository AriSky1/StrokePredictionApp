# pip install --pre pycaret


import numpy as np
import joblib
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report



df = pd.read_csv("data_strokes_prediction.csv")


# clean

df = df.drop(['id'], axis=1)  # drop useless column
df = df.dropna(axis=0)  # drop null values


# transform
df['stroke'] = df['stroke'].astype(str) # change type
le = LabelEncoder()
cat_cols = df[['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']]
for col in cat_cols:
       le.fit(df[col])
       df[col] = le.transform(df[col]) #encode categories

# X,y
X,y = df.drop('stroke', axis = 1), df['stroke']





#transform2
# sm = SMOTE(random_state=42)
# X_train_res, y_train_res = sm.fit_resample(X, y)
# print('Class distribution before resampling:', y.value_counts())
# print('Class distribution after resampling:', y_train_res.value_counts())
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

# split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)



# set model
# clf = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.2, n_estimators=100,
#                         max_depth=5, min_child_weight=3, max_delta_step=1, subsample=0.8)
clf = xgb.XGBClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
eval_set = [(X_res, y_res), (X_test, y_test)]
y_train_res = LabelEncoder().fit_transform(y_res)
y_test = le.fit_transform(y_test)
clf.fit(X_res, y_train_res, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)


#evaluation
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

joblib.dump(clf, "clf.pkl")