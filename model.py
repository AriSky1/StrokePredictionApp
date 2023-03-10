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
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score


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


# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#transform2
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print('Class distribution before resampling:', y_train.value_counts())
print('Class distribution after resampling:', y_train_res.value_counts())



# set model
clf = xgb.XGBClassifier()
eval_set = [(X_train_res, y_train_res), (X_test, y_test)]
le = LabelEncoder()
y_train_res = le.fit_transform(y_train_res)
y_test = le.fit_transform(y_test)
clf.fit(X_train_res, y_train_res, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)


#evaluation
y_pred = clf.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

joblib.dump(clf, "clf.pkl")