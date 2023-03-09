from flask import Flask, request, render_template
import pandas as pd
import joblib



app = Flask(__name__)

genders = ['Female', 'Male', 'Other']
hypertensions = ['Yes', 'No']
heart_diseases = ['Yes', 'No']
ever_marrieds = ['Yes', "No"]
work_types = ['A', 'B ']
Residence_types = ['A', 'B']
smoking_statuses = ['A', 'B']


@app.route('/', methods=['GET', 'POST'])
def main():
    # If a form is submitted
    if request.method == "POST":

        # Load model
        clf = joblib.load("clf.pkl")

        # Get values through input bars
        # id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke

        gender = request.form.get('gender')
        age = request.form.get("age")
        hypertension = request.form.get("hypertension")
        heart_disease = request.form.get("heart_disease")
        ever_married = request.form.get("ever_married")
        work_type= request.form.get("work_type")
        Residence_type= request.form.get("Residence_type")
        avg_glucose_level= request.form.get("avg_glucose_level")
        bmi= request.form.get("bmi")
        smoking_status= request.form.get("smoking_status")

        # Put inputs to dataframe
        # X = pd.DataFrame([[gender_female, gender_male, gender_other, age]],
        #                  columns=["gender_Female","gender_Male","gender_Other", "age"])

        # Get prediction
        # prediction = clf.predict(X)[0]
        prediction =gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status

    else:
        prediction = ""

    return render_template("website.html", genders=genders, hypertensions=hypertensions, heart_diseases=heart_diseases,
                           ever_marrieds=ever_marrieds, work_types=work_types, Residence_types=Residence_types,
                           smoking_statuses=smoking_statuses,output=prediction)




# @app.route("/test" , methods=['GET', 'POST'])
# def test():
#     select = request.form.get('comp_gender')
#     return(str(select)) # just to see what select is








# Running the app
if __name__ == '__main__':
    app.run(debug = True)