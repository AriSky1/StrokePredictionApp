

import numpy as np
import joblib
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler


from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report



df = pd.read_csv("data_strokes_prediction.csv")



# transform
le = LabelEncoder()

df = df.drop(['id'], axis=1)  # drop useless column
df = df.fillna(29.0) # fill null values with avg bmi

# cat_cols = df[['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke']]
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print(cat_cols)
for col in cat_cols:
       le.fit(df[col])
       df[col] = le.transform(df[col]) #encode categories



# X,y
X,y = df.drop('stroke', axis = 1), df['stroke']



# undersample
rus = RandomUnderSampler(random_state=42) #sample coan be biased so test many times without random state
# rus = RandomUnderSampler()
X_res, y_res = rus.fit_resample(X, y)

#oversample #OVERFITTING
# rus = RandomOverSampler() #sample coan be biased so test many times without random state
# # rus = RandomUnderSampler()
# X_res, y_res = rus.fit_resample(X, y)


#split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)


#model
clf = RandomForestClassifier(random_state=0)

#fit
clf.fit(X_train, y_train)

#evaluation
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

joblib.dump(clf, "clf.pkl")