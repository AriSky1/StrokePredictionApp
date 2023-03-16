from flask import Flask, request, render_template, flash
import pandas as pd
import pickle as pkl
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import pycaret
from pycaret.classification import *


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'




genders = ['Female', 'Male']
hypertensions = ['1', '0']
heart_diseases = ['1', '0']
ever_marrieds = ['Yes', 'No']
work_types = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
Residence_types = ['Urban', 'Rural']
smoking_statuses = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']





@app.route('/', methods=['GET', 'POST'])
def main():
    # If a form is submitted
    if request.method == "POST":

        prediction_label = None

    # Get user input


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


    # Encode user_input the same way as the training data was encoded before the training

        #create dataframe with user inputs
        usr_input =pd.DataFrame([[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]],
                                columns=['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status'])
        #load training data
        df = pd.read_csv(r'data_strokes_prediction.csv')
        #drop prediction column in order to be able to append usr_input row to training data
        df=df.drop(columns=['id','stroke'])
        # append in order to be able to encode all the dataset with the usr_input as the last row
        # and ensure encoding is the same
        df=df.append(usr_input)
        #encozsde all the dataset into categories
        le = LabelEncoder()
        cat_cols = df[['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']]
        for col in cat_cols:
            le.fit(df[col])
            df[col] = le.transform(df[col])

    # Isolate encoded user input

        # X are user inputs to predict, format: dataframe
        X = df.iloc[-1:]


    # Use trained model to predict 0 or 1 on the user input

        clf = pkl.load(open('best_pipeline1.pkl', 'rb'))

        prediction_label = clf.predict(X)
        prediction_label = int(prediction_label)
        print(prediction_label, type(prediction_label))
        prediction_score = clf.predict_proba(X)
        print(prediction_score)



    # Set the prediction message to the user


        if prediction_label == 0:
            prediction = ' On ' + str(round(prediction_score[0][0] * 100)) + ' % sure there is low risk.'
            return render_template("website.html", genders=genders, hypertensions=hypertensions,
                                   heart_diseases=heart_diseases,
                                   ever_marrieds=ever_marrieds, work_types=work_types, Residence_types=Residence_types,
                                   smoking_statuses=smoking_statuses, output=prediction)
        if prediction_label == 1:
            prediction = ' On '+ str(round(prediction_score[0][1] * 100))+' % sure there is high risk.'
            return render_template("website.html", genders=genders, hypertensions=hypertensions, heart_diseases=heart_diseases,
                               ever_marrieds=ever_marrieds, work_types=work_types, Residence_types=Residence_types,
                               smoking_statuses=smoking_statuses,output=prediction)

        else:
            prediction = ""
            return render_template("website.html", genders=genders, hypertensions=hypertensions, heart_diseases=heart_diseases,
                               ever_marrieds=ever_marrieds, work_types=work_types, Residence_types=Residence_types,
                               smoking_statuses=smoking_statuses,output=prediction)





    return render_template("website.html", genders=genders, hypertensions=hypertensions, heart_diseases=heart_diseases,
                       ever_marrieds=ever_marrieds, work_types=work_types, Residence_types=Residence_types,
                       smoking_statuses=smoking_statuses)










# Running the app
if __name__ == '__main__':
    app.run(debug = True)
