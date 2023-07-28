import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from imblearn.under_sampling import RandomUnderSampler
pd.options.mode.chained_assignment = None  # default='warn'

random_state = 42


def transformation_pipeline(df):

    # drop useless
    df = df[(df['gender'] != 'Other')]
    df = df.drop(['id'], axis=1)


    df['gender']=df['gender'].astype(str)
    df['age']=df['age'].astype(float)
    df['hypertension']=df['hypertension'].astype(int)
    df['heart_disease']=df['heart_disease'].astype(int)
    df['ever_married']=df['ever_married'].astype(str)
    df['work_type']=df['work_type'].astype(str)
    df['Residence_type']=df['Residence_type'].astype(str)
    df['avg_glucose_level']=df['avg_glucose_level'].astype(float)
    df['bmi']=df['bmi'].astype(float)
    df['smoking_status']=df['smoking_status'].astype(str)


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


    df['heart_disease']=df['heart_disease'].astype(str)
    df['hypertension']=df['hypertension'].astype(str)

    df.hypertension = df.hypertension.replace({'Yes': 1, 'No': 0, '': -1}).astype(np.uint8)
    df.heart_disease = df.heart_disease.replace({'Yes': 1, 'No': 0, '': -1}).astype(np.uint8)
    df.ever_married = df.ever_married.replace({'Yes': 1, 'No': 0, '': -1}).astype(np.uint8)
    df.work_type = df.work_type.replace(
        {'Private': 2, 'Govt_job': 0, 'Self-employed': 3, 'children': 4, 'Never_worked': 1, '': -1}).astype(np.uint8)
    df.Residence_type = df.Residence_type.replace({'Urban': 1, 'Rural': 0, '': -1}).astype(np.uint8)
    df.smoking_status = df.smoking_status.replace(
        {'never smoked': 2, 'smokes': 3, 'Unknown': 0, 'formerly smoked': 1, '': -1}).astype(np.uint8)
    df.gender = df.gender.replace({'Male': 0, 'Female': 1, 'Other': -1, '': -1}).astype(np.uint8)
    df.avg_glucose_level = df.avg_glucose_level.astype(np.float64)
    df.bmi = df.bmi.astype(np.float64)
    df.age = df.age.astype(np.float64)

    # df.to_csv('df_encoded.csv')
    # create X,y
    X,y = df.drop('stroke', axis = 1), df['stroke']

    # undersampling + oversampling
    over = SMOTE(sampling_strategy = 1)
    under = RandomUnderSampler(sampling_strategy = 0.1)
    X_res_u, y_res_u = under.fit_resample(X, y)
    X_res, y_res = over.fit_resample(X_res_u, y_res_u)
    print('X_res shape\n', X_res.shape)
    print('y_res shape\n', y_res.shape)

    print('y_res balance\n', y_res.value_counts())

    return X_res, y_res



def transformation_pipeline_nores(df):

    # drop useless
    df = df[(df['gender'] != 'Other')]
    df = df.drop(['id'], axis=1)

    df['gender'] = df['gender'].astype(str)
    df['age'] = df['age'].astype(float)
    df['hypertension'] = df['hypertension'].astype(int)
    df['heart_disease'] = df['heart_disease'].astype(int)
    df['ever_married'] = df['ever_married'].astype(str)
    df['work_type'] = df['work_type'].astype(str)
    df['Residence_type'] = df['Residence_type'].astype(str)
    df['avg_glucose_level'] = df['avg_glucose_level'].astype(float)
    df['bmi'] = df['bmi'].astype(float)
    df['smoking_status'] = df['smoking_status'].astype(str)

    # fill null values with regression on numerical values
    DT_bmi_pipe = Pipeline(steps=[
        ('scale', StandardScaler()),
        ('lr', DecisionTreeRegressor(random_state=random_state))
    ])
    X = df[['age', 'gender', 'bmi']].copy()
    X.gender = X.gender.replace({'Male': 0, 'Female': 1, 'Other': -1}).astype(np.uint8)
    Missing = X[X.bmi.isna()]
    X = X[~X.bmi.isna()]
    Y = X.pop('bmi')
    DT_bmi_pipe.fit(X, Y)
    predicted_bmi = pd.Series(DT_bmi_pipe.predict(Missing[['age', 'gender']]), index=Missing.index)
    df.loc[Missing.index, 'bmi'] = round(predicted_bmi)

    df['heart_disease'] = df['heart_disease'].astype(str)
    df['hypertension'] = df['hypertension'].astype(str)

    df.hypertension = df.hypertension.replace({'Yes': 1, 'No': 0, '': -1}).astype(np.uint8)
    df.heart_disease = df.heart_disease.replace({'Yes': 1, 'No': 0, '': -1}).astype(np.uint8)
    df.ever_married = df.ever_married.replace({'Yes': 1, 'No': 0, '': -1}).astype(np.uint8)
    df.work_type = df.work_type.replace(
        {'Private': 2, 'Govt_job': 0, 'Self-employed': 3, 'children': 4, 'Never_worked': 1, '': -1}).astype(
        np.uint8)
    df.Residence_type = df.Residence_type.replace({'Urban': 1, 'Rural': 0, '': -1}).astype(np.uint8)
    df.smoking_status = df.smoking_status.replace(
        {'never smoked': 2, 'smokes': 3, 'Unknown': 0, 'formerly smoked': 1, '': -1}).astype(np.uint8)
    df.gender = df.gender.replace({'Male': 0, 'Female': 1, 'Other': -1, '': -1}).astype(np.uint8)
    df.avg_glucose_level = df.avg_glucose_level.astype(np.float64)
    df.bmi = df.bmi.astype(np.float64)
    df.age = df.age.astype(np.float64)

        # df.to_csv('df_encoded.csv')
        # create X,y
    X, y = df.drop('stroke', axis=1), df['stroke']

        # undersampling + oversampling
        # over = SMOTE(sampling_strategy=1)
        # under = RandomUnderSampler(sampling_strategy=0.1)
        # X_res_u, y_res_u = under.fit_resample(X, y)
        # X_res, y_res = over.fit_resample(X_res_u, y_res_u)
        # print('X_res shape\n', X_res.shape)
        # print('y_res shape\n', y_res.shape)

        # print('y_res balance\n', y_res.value_counts())

    return X, y



df = pd.read_csv(r'data_strokes_prediction.csv')
transformation_pipeline(df)