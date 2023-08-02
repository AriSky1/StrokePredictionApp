from flask import Flask, request,  render_template
import pickle as pkl
import pandas as pd
import numpy as np



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

        pipeline = pkl.load(open('model.pkl', 'rb'))

        prediction_label = pipeline.predict(d)
        prediction_label = int(prediction_label)
        prediction_score = pipeline.predict_proba(d)


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



if __name__ == '__main__':
    app.run(debug = True)
