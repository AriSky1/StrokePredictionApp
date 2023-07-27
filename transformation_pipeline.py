import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from imblearn.under_sampling import RandomUnderSampler


random_state = 42


def transformation_pipeline(df):

    # drop useless
    df = df[(df['gender'] != 'Other')]
    df = df.drop(['id'], axis=1)

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
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
           le.fit(df[col])
           df[col] = le.transform(df[col])

    # create X,y
    X,y = df.drop('stroke', axis = 1), df['stroke']

    # undersampling + oversampling
    over = SMOTE(sampling_strategy = 1)
    under = RandomUnderSampler(sampling_strategy = 0.1)
    X_res_u, y_res_u = under.fit_resample(X, y)
    X_res, y_res = over.fit_resample(X_res_u, y_res_u)

    return X_res, y_res