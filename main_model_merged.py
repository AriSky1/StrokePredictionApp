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
from flask import Flask, request, make_response, render_template, flash
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from transformation_pipeline import transformation_pipeline, transformation_pipeline_nores
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from imblearn.under_sampling import RandomUnderSampler
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





app = Flask(__name__)



genders = ['Female', 'Male']
hypertensions = ['Yes', 'No']
heart_diseases = ['Yes', 'No']
ever_marrieds = ['Yes', 'No']
work_types = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never worked']
Residence_types = ['Urban', 'Rural']
smoking_statuses = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']


@app.route('/', methods=['GET', 'POST'])



def main():


    if request.method == "POST":

        gender = request.form.get('gender')
        age = request.form.get("age")
        hypertension = request.form.get("hypertension")
        heart_disease = request.form.get("heart_disease")
        ever_married = request.form.get("ever_married")
        work_type = request.form.get('work_type')
        Residence_type = request.form.get("Residence_type")
        avg_glucose_level = request.form.get("avg_glucose_level")
        bmi = request.form.get("bmi")
        smoking_status = request.form.get('smoking_status')


        random_state = 42
        df = pd.read_csv("data_strokes_prediction.csv")
        X_res, y_res = transformation_pipeline(df)
        print('X_res shape\n',X_res.shape)
        print('y_res shape\n', y_res.shape)

        print('y_res balance\n', y_res.value_counts())


        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.33, random_state=random_state)


        print('X_test shape', X_test.shape)
        print('y_test shape', y_test.shape)
        usr_input = {'gender': gender, 'age': age, 'hypertension': hypertension, 'heart_disease': heart_disease, 'ever_married': ever_married, 'work_type': work_type, 'Residence_type': Residence_type, 'avg_glucose_level': avg_glucose_level, 'bmi': bmi, 'smoking_status' :smoking_status}



        s = pd.Series(usr_input, index=usr_input.keys())
        d = pd.DataFrame(s).transpose()


        d.hypertension = d.hypertension.replace({'Yes': 1, 'No': 0,'': -1}).astype(np.uint8)
        d.heart_disease = d.heart_disease.replace({'Yes': 1, 'No': 0, '': -1}).astype(np.uint8)
        d.ever_married = d.ever_married.replace({'Yes': 1, 'No': 0, '': -1}).astype(np.uint8)
        d.work_type = d.work_type.replace({'Private': 2, 'Govt_job': 0, 'Self-employed': 3, 'children': 4, 'Never_worked': 1, '': -1}).astype(np.uint8)
        d.Residence_type = d.Residence_type.replace({'Urban': 1, 'Rural': 0, '': -1}).astype(np.uint8)
        d.smoking_status = d.smoking_status.replace({'never smoked': 2, 'smokes': 3, 'Unknown': 0, 'formerly smoked':1, '': -1 }).astype(np.uint8)
        d.gender = d.gender.replace({'Male': 0, 'Female': 1, 'Other': -1, '': -1}).astype(np.uint8)
        d.Residence_type = d.Residence_type.replace({'Urban': 1, 'Rural': 0, '': -1}).astype(np.uint8)
        d.avg_glucose_level = d.avg_glucose_level.astype(np.float64)
        d.bmi = d.bmi.astype(np.float64)
        d.age = d.age.astype(np.float64)

        print('d is the variable to be predicted\n',d)
        print('d shape\n',d.shape)
        print('d dtypes\n',d.dtypes)
        print('d type\n',type(d))

        random_state = 42
        XGB_BEST_PARAMS = {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 180}

        xg = XGBClassifier(params=XGB_BEST_PARAMS, random_state=42)

        # Fit the training data to the model
        xg = xg.fit(X_train, y_train)

        # Get cv scores
        cross_val_scores = cross_val_score(xg, X_res, y_res, cv=5)
        print("Cross-Validation Scores:", cross_val_scores)
        print("Mean Cross-Validation Score:", np.mean(cross_val_scores))
        # pipeline = Pipeline(steps=[('scale', StandardScaler()),
        #                             ("XGBC", XGBClassifier(random_state=random_state, params=XGB_BEST_PARAMS))])


        pipeline = Pipeline(steps=[('scale', StandardScaler()),
                                    ("XGBC", LogisticRegression(random_state=random_state, C=10, penalty='l1', solver='liblinear'))])
        pipeline.fit(X_train, y_train)


        prediction_label = pipeline.predict(d)
        prediction_label = int(prediction_label)
        prediction_score = pipeline.predict_proba(d)

        print('PREDS :', prediction_label)

        dt = pd.read_csv('data_test.csv')
        print(d)
        # dt.hypertension = dt.hypertension.replace({'Yes': 1, 'No': 0,'': -1}).astype(np.uint8)
        # dt.heart_disease = dt.heart_disease.replace({'Yes': 1, 'No': 0, '': -1}).astype(np.uint8)
        # dt.ever_married = dt.ever_married.replace({'Yes': 1, 'No': 0, '': -1}).astype(np.uint8)
        # dt.work_type = dt.work_type.replace({'Private': 2, 'Govt_job': 0, 'Self-employed': 3, 'children': 4, 'Never_worked': 1, '': -1}).astype(np.uint8)
        # dt.Residence_type = dt.Residence_type.replace({'Urban': 1, 'Rural': 0, '': -1}).astype(np.uint8)
        # dt.smoking_status = dt.smoking_status.replace({'never smoked': 2, 'smokes': 3, 'Unknown': 0, 'formerly smoked':1, '': -1 }).astype(np.uint8)
        # dt.gender = dt.gender.replace({'Male': 0, 'Female': 1, 'Other': -1, '': -1}).astype(np.uint8)
        # dt.Residence_type = dt.Residence_type.replace({'Urban': 1, 'Rural': 0, '': -1}).astype(np.uint8)
        # dt.avg_glucose_level = dt.avg_glucose_level.astype(np.float64)
        # dt.bmi = dt.bmi.astype(np.float64)
        # dt.age = dt.age.astype(np.float64)
        # labels = dt['stroke']
        # dt = dt.drop(columns=['id', 'stroke'], axis=1)

        X_res,y_res=transformation_pipeline_nores(dt)
        print('d\n',d)
        print(dt.shape)




        prediction_label_dt = pipeline.predict(X_res)
        print('prediction_label_dt',prediction_label_dt)

        print('y_test_shape\n', y_test.shape)

        print(classification_report(y_res, prediction_label_dt))
        print('Accuracy Score: ', accuracy_score(y_res, prediction_label_dt))
        print('Recall Score: ', recall_score(y_res, prediction_label_dt))
        print('\nConfusion matrix: \n', confusion_matrix(y_res, prediction_label_dt))
        print('\n Model :', 'XGB')


        if prediction_label == 0:
            prediction = ' On ' + str(prediction_score[0][0] * 100)[0:5] + ' % sure there is low risk.'
            color1 = '#1cc78b'
            return render_template("website.html", genders=genders, hypertensions=hypertensions,
                                   heart_diseases=heart_diseases,
                                   ever_marrieds=ever_marrieds, work_types=work_types, Residence_types=Residence_types,
                                   smoking_statuses=smoking_statuses, output=prediction, color=color1)

        if prediction_label == 1:
            prediction = ' On ' + str(prediction_score[0][1] * 100)[0:5] + ' % sure there is high risk.'
            color2 = '#b3505a'
            return render_template("website.html", genders=genders, hypertensions=hypertensions, heart_diseases=heart_diseases,
                               ever_marrieds=ever_marrieds, work_types=work_types, Residence_types=Residence_types,
                               smoking_statuses=smoking_statuses,output=prediction, color=color2)

        else:
            prediction = 'Enter data'
            return render_template("website.html", genders=genders, hypertensions=hypertensions, heart_diseases=heart_diseases,
                               ever_marrieds=ever_marrieds, work_types=work_types, Residence_types=Residence_types,
                               smoking_statuses=smoking_statuses,output=prediction)


    if request.method == "GET":

        prediction = ""




    return render_template("website.html", genders=genders, hypertensions=hypertensions, heart_diseases=heart_diseases,
                               ever_marrieds=ever_marrieds, work_types=work_types, Residence_types=Residence_types,
                               smoking_statuses=smoking_statuses, prediction=prediction)



# Running the app
if __name__ == '__main__':
    app.run(debug = True)
