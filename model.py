import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pycaret.classification import *
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

#load data
df = pd.read_csv("data_strokes_prediction.csv")


# fill null values with regression on numerical values
DT_bmi_pipe = Pipeline( steps=[
                               ('scale',StandardScaler()),
                               ('lr',DecisionTreeRegressor(random_state=42))
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

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# augment : artificially get the same amount of stroke and non stroke cases for perfect balance
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# setup
train_data = pd.concat([X_train_res, y_train_res], axis=1)
s = setup(data=train_data, target='stroke', session_id=123, normalize=True)

# grab best model
best = compare_models()

# check metrics

# xg = create_model('xgboost')
# tuned_xg = tune_model(xg)
# best=tuned_xg
# plot_model(xg)
# plot_model(xg, plot = 'error')
# plot_model(xg, plot = 'confusion_matrix')
#save model to pkl format
save_model(best, 'best_pipeline')


