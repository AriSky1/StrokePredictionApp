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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


random_state = 42




#load data


df = pd.read_csv("data_strokes_prediction.csv")

df=df.drop(df[df['id'] == 69768].index)
df=df.drop(df[df['id'] == 49669].index)

# df=df.dropna(axis=0)
#drop gender other
# df = df[(df['gender'] != 'Other')]

# drop outliers in age : 69768 (stroke at 1.32), 49669 (stroke at 14)
# outlier1 = df.loc[df['id'] == 69768]
# outlier2 = df.loc[df['id'] == 49669]
# print('outlier1',outlier1)
# print('outlier2',outlier2)


# transform : encode types object into categories
le = LabelEncoder()

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
       le.fit(df[col])
       df[col] = le.transform(df[col])

df= df[['age', 'bmi', 'avg_glucose_level', 'heart_disease', 'stroke']]
print('Missing values in bmi before regression',df['bmi'].isnull().values.any())





# fill null values with regression on numerical values
DT_bmi_pipe = Pipeline( steps=[
                               ('scale',StandardScaler()),
                               ('lr',DecisionTreeRegressor(random_state=random_state))
                              ])
X = df[['age','bmi','avg_glucose_level', 'heart_disease']].copy()
Missing = X[X.bmi.isna()]
X = X[~X.bmi.isna()]
Y = X.pop('bmi')
DT_bmi_pipe.fit(X,Y)
predicted_bmi = pd.Series(DT_bmi_pipe.predict(Missing[['age','avg_glucose_level', 'heart_disease']]),index=Missing.index)
df.loc[Missing.index,'bmi'] = round(predicted_bmi)

print('Missing values in bmi after regression :  ',df['bmi'].isnull().values.any())
print('Missing remplaced', df.loc[Missing.index,'bmi'])


# df=pd.DataFrame(cat_cols, columns=['age', 'work_type', 'ever_married', 'heart_disease', 'stroke'])

# # # # feature selection
# df=df[['age', 'avg_glucose_level', 'bmi', 'stroke']]
# # define X as features, y as labels to predict
X,y = df.drop('stroke', axis = 1), df['stroke']
print(X,y)
# undersampling + oversampling
over = SMOTE(sampling_strategy = 1)

under = RandomUnderSampler(sampling_strategy = 0.1)

# print(X.shape,y.shape)
X_res_u, y_res_u = under.fit_resample(X, y)
# print(y_res_u.value_counts())
# print(X_res_u.shape,y_res_u.shape)


# print(y_res_u)
X_res, y_res = over.fit_resample(X_res_u, y_res_u)
# print(y_res.value_counts())
# print(X_res.shape,y_res.shape)

# print('X_res',X_res)


#split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=random_state)



# # Hyperparameter tuning using GridSearch
# estimator = XGBClassifier(objective='binary:logistic', nthread=4, seed=random_state)
# parameters = {
#     'max_depth': range(2, 10, 1),
#     'n_estimators': range(60, 220, 40),
#     'learning_rate': [0.1, 0.01, 0.05]
# }
# grid_search = GridSearchCV(estimator=estimator, param_grid=parameters, scoring='roc_auc', n_jobs=10, cv=10, verbose=True)
# grid_search.fit(X_train, y_train)
#
# # Get the best hyperparameters from GridSearch
# best_params = grid_search.best_params_
# print("Best Hyperparameters:", best_params)



# xg = XGBClassifier(n_estimators=180, learning_rate=0.1, max_depth=9,random_state=42)
#
#
# # Fit the model to the training data
# xg.fit(X_train, y_train)
#
# # Get feature importances
# feature_importances = xg.feature_importances_
#
# # Create a DataFrame for visualization
# importances_df = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': feature_importances
# })
#
# # Sort the DataFrame by importance in descending order
# importances_df = importances_df.sort_values(by='Importance', ascending=False)
#
# # Plotting feature importances
# plt.figure(figsize=(10, 8))
# sns.barplot(data=importances_df, x='Importance', y='Feature', color='b')
# plt.title('Feature Importance XGBoost')
# plt.xlabel('Importance')
# plt.ylabel('Feature')
# plt.show()
#
#






# train
model,name = XGBClassifier(),'XGBC'
# XGB_BEST_PARAMS = {'colsample_bytree': 0.6966652866539669,
#                    'gamma': 5.1222111958093395,
#                    'learning_rate': 0.28,
#                    'max_depth': 16.0,
#                    'min_child_weight': 1.0,
#                    'nthread': 3.0, 'reg_alpha': 61.0,
#                    'reg_lambda': 0.5855571439718152,
#                    'scale_pos_weight': 4.0, 'subsample': 0.9}
# XGB_BEST_PARAMS = {
#                    'learning_rate': 0.28,
# }
# XGB_BEST_PARAMS = {
# 'learning_rate': 0.1, 'max_depth': 8, 'n_estimators': 180
# }


XGB_BEST_PARAMS = {
'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 180
}


# XGB_BEST_PARAMS = {
#
# }


# cross_val_scores = cross_val_score(model, X, y, cv=5)
#
# # Print the cross-validation scores for each fold
# print("Cross-Validation Scores:", cross_val_scores)
# print("Mean Cross-Validation Score:", np.mean(cross_val_scores))







pipeline = Pipeline(steps = [('scale',StandardScaler()), ("XGBC",XGBClassifier(random_state=random_state, params=XGB_BEST_PARAMS))])




pipeline.fit(X_train,y_train)

# print('X_train',X_train)


input= pd.read_csv('data_test.csv')
input= input[['age', 'bmi', 'avg_glucose_level', 'heart_disease', 'stroke']]
# print(input)

tuned_pred = pipeline.predict(X_test)
print('tuned_pred',tuned_pred)






input = input. iloc[:,:-1]
# print('input',input)
cat_cols = input[['bmi']]
for col in cat_cols:
    le.fit(input[col])
    input[col] = le.transform(input[col])
# print('input',input)
my_pred = pipeline.predict(input)
print('PREDS :' , my_pred)




print(classification_report(y_test,tuned_pred))
print('Accuracy Score: ',accuracy_score(y_test,tuned_pred))
print('Recall Score: ',recall_score(y_test,tuned_pred))
print('\nConfusion matrix: \n',confusion_matrix(y_test, tuned_pred))
print('\n Model :',name)




# print('y_test',y_test, 'tuned_pred',tuned_pred)
comp = pd.DataFrame()
comp['true'] = y_test
comp['predicted'] = tuned_pred
comp['diff'] = comp['true'] - comp['predicted']
print(comp['diff'].value_counts())
# pkl.dump(pipeline, open('best_pipeline1.pkl', 'wb'))