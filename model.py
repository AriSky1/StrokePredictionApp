import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
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
from mlxtend.classifier import StackingCVClassifier
from sklearn.svm import SVC
from transformation_pipeline import transformation_pipeline

random_state = 42


#load data
df = pd.read_csv("data_strokes_prediction.csv")

# transform data
X_res,y_res = transformation_pipeline(df)

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

XGB_BEST_PARAMS = {
'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 180
}



xg = XGBClassifier(n_estimators=180, learning_rate=0.1, max_depth=9,random_state=42)


# Fit the model to the training data
xg = xg.fit(X_train, y_train)

# Get cv scores
cross_val_scores = cross_val_score(xg, X_res, y_res, cv=5)
print("Cross-Validation Scores:", cross_val_scores)
print("Mean Cross-Validation Score:", np.mean(cross_val_scores))


# Get feature importances
feature_importances = xg.feature_importances_

# Create a DataFrame for visualization
importances_df = pd.DataFrame({
    'Feature': X_res.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by importance in descending order
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(10, 8))
sns.barplot(data=importances_df, x='Importance', y='Feature', color='b')
plt.title('Feature Importance XGBoost')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()



# train

pipeline = Pipeline(steps = [('scale',StandardScaler()), ("XGBC",XGBClassifier(random_state=random_state, params=XGB_BEST_PARAMS))])
pipeline.fit(X_train,y_train)


# predict
tuned_pred = pipeline.predict(X_test)


# print report
print(classification_report(y_test,tuned_pred))
print('Accuracy Score: ',accuracy_score(y_test,tuned_pred))
print('Recall Score: ',recall_score(y_test,tuned_pred))
print('\nConfusion matrix: \n',confusion_matrix(y_test, tuned_pred))
print('\n Model :','XGBoost')



# VERIFICATION BY 20 UNSEEN ROWS (10 first P & 10 last N)


print('VERIFICATION BY 20 UNSEEN ROWS (10 first P & 10 last N) :  \n')
input= pd.read_csv('data_test.csv')

input = input. iloc[:,:-1]
input=input.drop(columns=input.columns[0], axis=1)

le = LabelEncoder()
cat_cols = input[['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']]
for col in cat_cols:
    le.fit(input[col])
    input[col] = le.transform(input[col])

# print('input',input)
my_pred = pipeline.predict(input)
print('PREDS :' , my_pred)

# export model
pkl.dump(pipeline, open('best_pipeline.pkl', 'wb'))