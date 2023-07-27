
import pandas as pd
import numpy as np
import time
# import eli5
from sklearn import *

# model algorithams
from sklearn.ensemble import (RandomForestClassifier,
                              AdaBoostClassifier,
                              GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.svm import SVC, LinearSVC
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from feature_engine.wrappers import SklearnTransformerWrapper

#Common model helpers
from sklearn.preprocessing import (StandardScaler,
                                   LabelEncoder,
                                   OneHotEncoder)
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,
                             auc,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             confusion_matrix)
from sklearn.model_selection import (GridSearchCV,
                                     StratifiedKFold,
                                     cross_val_score)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pycaret.classification import *
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import shuffle
from IPython.display import display



random_state=42

XGB_BEST_PARAMS = {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 180}
# LG_BEST_PARAMS ={'C': 4.635, 'penalty': 'l2', 'tol': 0.0001}
LR_BEST_PARAMS = {}
# SVC_BEST_PARAMS = {'kernel': 'rbf', 'gamma': 0.001, 'C': 1000}
SVC_BEST_PARAMS = {}



#load data
df = pd.read_csv("data_strokes_prediction.csv")

#drop gender other
df = df[(df['gender'] != 'Other')]

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
df = df.drop(['id'], axis=1)  # drop useless column
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
       le.fit(df[col])
       df[col] = le.transform(df[col])



X,y = df.drop('stroke', axis = 1), df['stroke']


# undersampling + oversampling
over = SMOTE(sampling_strategy = 1)
under = RandomUnderSampler(sampling_strategy = 0.1)
X_res_u, y_res_u = under.fit_resample(X, y)
X_res, y_res = over.fit_resample(X_res_u, y_res_u)

#split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=random_state)



def test_main_classifiers(x_set, y_set):
    t1 = time.time()
    print('Classification Process Starts....')
    accuracy, precision, recall, f1, auc, conf_mat = [], [], [], [], [], []

    random_state = None

    ##classifiers list
    classifiers = []
    classifiers.append(SVC(random_state=random_state, probability=True).set_params(**SVC_BEST_PARAMS))
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state)))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    #     classifiers.append(LogisticRegression(random_state=random_state))
    classifiers.append(LogisticRegression(random_state=random_state).set_params(**LR_BEST_PARAMS))
    #     classifiers.append(XGBClassifier(random_state=random_state))
    classifiers.append(XGBClassifier(params=XGB_BEST_PARAMS))
    classifiers.append(LGBMClassifier(random_state=random_state, learning_rate=0.067))

    for classifier in classifiers:
        t = time.time()
        print('fitting on classifier with parameters: {}'.format(classifier))

        # classifier and fitting
        clf = classifier
        clf.fit(x_set, y_set)

        # predictions
        y_preds = clf.predict(X_test)
        y_probs = clf.predict_proba(X_test)

        # metrics
        accuracy.append((round(accuracy_score(y_test, y_preds), 2)) * 100)
        precision.append((round(precision_score(y_test, y_preds), 2)) * 100)
        recall.append((round(recall_score(y_test, y_preds), 2)) * 100)
        f1.append((round(f1_score(y_test, y_preds), 2)) * 100)
        auc.append((round(roc_auc_score(y_test, y_probs[:, 1]), 2)) * 100)
        conf_mat.append(confusion_matrix(y_test, y_preds))

        elapsed = time.time() - t
        print('Done and elapsed time is {}seconds'.format(round(elapsed, 3)))
        print('\n')

    results_df = pd.DataFrame({"Accuracy Score": accuracy, "Precision Score": precision,
                               "Recall Score": recall, "f1 Score": f1, "AUC Score": auc,
                               "Confusion Matrix": conf_mat,
                               "Algorithm": ["SVC", "DecisionTree", "AdaBoost",
                                             "RandomForest", "GradientBoosting",
                                             "KNeighboors", "LogisticRegression",
                                             "XGBoost", "LightGBM"]})

    results_df = (results_df.sort_values(by='Algorithm', ascending=False)
                  .reset_index(drop=True))
    t2 = time.time() - t1
    print('\nClassification is Completed and results are strored in dataframe.\ntotal time elapsed is {}seconds'.format(
        t2))
    print('***************************************************************\n\n')

    return results_df


resamp_results=test_main_classifiers(X_train, y_train)
resamp_results.sort_values(by='f1 Score', ascending=False)
print(type(resamp_results))

with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(resamp_results)
