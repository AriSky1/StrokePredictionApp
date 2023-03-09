import pandas as pd
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("data_strokes_prediction.csv")


# code transformation

# df['gender'] =  df['gender'].replace(['Male', 'Female'], [0,1])
df = pd.get_dummies(df)

# print(df)

X = df[["gender_Female", "gender_Male", "gender_Other", "age"]]
y = df["stroke"]
#
clf = GaussianNB()
clf.fit(X, y)


import joblib

joblib.dump(clf, "clf.pkl")