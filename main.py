from flask import Flask, request, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)
from flask_pretty import Prettify
prettify = Prettify()
prettify.init_app(app)

genders = ['Female', 'Male', 'Other']
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

        # Load model
        clf = joblib.load("clf.pkl")

        # Get values through input bars
        # id,gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status,stroke

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



        # Encode


        usr_input =pd.DataFrame([[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]],
                                columns=['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status'])
        df = pd.read_csv(r'data_strokes_prediction.csv')
        df=df.drop(columns=['id','stroke'])
        df=df.append(usr_input)
        # print(df.tail(1)) # user answer is the last row
        le = LabelEncoder()
        cat_cols = df[['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']]
        for col in cat_cols:
            le.fit(df[col])
            df[col] = le.transform(df[col])  # encode categories

        print(df,df.tail())


        # user input is X

        X = df.tail(1)

        X['age']=int(X['age'])
        X['hypertension'] = int(X['hypertension'])
        X['heart_disease'] = int(X['heart_disease'])
        X['avg_glucose_level'] = float(X['avg_glucose_level'])
        X['bmi'] = float(X['bmi'])



        # Get prediction
        prediction = clf.predict(X)[0]
        # prediction = clf.predict(X)
        if prediction == 0:
            prediction = 'Low risk.'
        if prediction == 1:
            prediction = 'High stroke risk !'



    else:
        prediction = ""

    return render_template("website.html", genders=genders, hypertensions=hypertensions, heart_diseases=heart_diseases,
                           ever_marrieds=ever_marrieds, work_types=work_types, Residence_types=Residence_types,
                           smoking_statuses=smoking_statuses,output=prediction)





# Running the app
if __name__ == '__main__':
    app.run(debug = True)

