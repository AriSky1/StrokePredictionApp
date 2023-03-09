import pandas as pd
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("data_strokes_prediction.csv")


# code transformation

df = pd.get_dummies(df, columns=['gender', 'ever_married','work_type', 'Residence_type', 'smoking_status'])
df=df.dropna(axis=0)
y = df["stroke"]
df=df.drop(['id', 'stroke'],axis=1)
print(df)
print(df.columns)

# print(df)

X = df[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
       'gender_Female', 'gender_Male', 'gender_Other', 'ever_married_No',
       'ever_married_Yes', 'work_type_Govt_job', 'work_type_Never_worked',
       'work_type_Private', 'work_type_Self-employed', 'work_type_children',
       'Residence_type_Rural', 'Residence_type_Urban',
       'smoking_status_Unknown', 'smoking_status_formerly smoked',
       'smoking_status_never smoked', 'smoking_status_smokes']]

#
clf = GaussianNB()
clf.fit(X, y)


import joblib

joblib.dump(clf, "clf.pkl")