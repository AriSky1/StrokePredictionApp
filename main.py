from flask import Flask, request, make_response, render_template, flash
import pandas as pd
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from transformation_pipeline import transformation_pipeline
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from imblearn.under_sampling import RandomUnderSampler

app = Flask(__name__)




# @app.errorhandler(ValueError)
# def value_error(e):
#     return "A value error occurred!"



genders = ['Female', 'Male']
hypertensions = ['Yes', 'No']
heart_diseases = ['Yes', 'No']
ever_marrieds = ['Yes', 'No']
work_types = ['Private', 'Self-employed', 'Government job', 'Children', 'Never worked']
Residence_types = ['Urban', 'Rural']
smoking_statuses = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']

random_state=42


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


        # usr_input =pd.DataFrame([[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]],
        #                         columns=['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status'])


        df = pd.read_csv(r'data_strokes_prediction.csv')
        test_df = pd.read_csv('data_test.csv')
        print('test_df', test_df)

        df=df.drop(columns=['id'])
        test_df = test_df.drop(columns=['id'])


        # usr_input =pd.DataFrame([gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status], index=df.columns )

        # Dictionary of columns
        usr_input = {'gender': gender, 'age': age, 'hypertension': hypertension, 'heart_disease': heart_disease, 'ever_married': ever_married, 'work_type': work_type, 'Residence_type': Residence_type, 'avg_glucose_level': avg_glucose_level, 'bmi': bmi, 'smoking_status' :smoking_status}

        usr_input_test = test_df.iloc[0].to_dict()
        usr_input_test1 = test_df.iloc[1].to_dict()
        usr_input_test2 = test_df.iloc[2].to_dict()
        usr_input_test3 = test_df.iloc[3].to_dict()
        usr_input_test4= test_df.iloc[4].to_dict()


        print('test_df', test_df)
        # usr_input = pd.DataFrame.from_dict(usr_input)
        # print('usr_input',usr_input)
        # print('type_usr_input',type(usr_input))

        s = pd.Series(usr_input, index=usr_input.keys())
        ss = pd.Series(usr_input_test, index=usr_input_test.keys())
        print('s      \n', s)


        d = pd.DataFrame(s).transpose()



        df=df.append(d)




        # print("df types after append :         \n", df.dtypes)
        print("df tail after append:         \n", df.tail(5))
        # df['heart_disease']=df['heart_disease'].astype(str)
        # df['hypertension']=df['hypertension'].astype(str)
        # print('usr_input : ', usr_input)



        df['gender'] = df['gender'].astype(str)
        df['age'] = df['age'].astype(float)
        # df['hypertension'] = df['hypertension'].astype(int)
        df['hypertension'] = df['hypertension'].astype(str)
        # df['heart_disease'] = df['heart_disease'].astype(int)
        df['heart_disease'] = df['heart_disease'].astype(str)
        df['ever_married'] = df['ever_married'].astype(str)
        df['work_type'] = df['work_type'].astype(str)
        df['Residence_type'] = df['Residence_type'].astype(str)
        df['avg_glucose_level'] = df['avg_glucose_level'].astype(float)
        df['bmi'] = df['bmi'].astype(float)
        df['smoking_status'] = df['smoking_status'].astype(str)



        le = LabelEncoder()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in cat_cols:
            le.fit(df[col])
            df[col] = le.transform(df[col])






        # create X,y
        X, y = df.drop('stroke', axis=1), df['stroke']




        topredict = X.iloc[-1:]




        clf = pkl.loads(open('best_pipeline1.pkl', 'rb'))


        prediction_label = clf.predict(topredict)
        prediction_label = int(prediction_label)
        # print(prediction_label, type(prediction_label))
        prediction_score = clf.predict_proba(topredict)
        # print(prediction_score)




        if prediction_label == 0:
            # prediction = ' On ' + str(round(prediction_score[0][0] * 100)) + ' % sure there is low risk.'
            prediction = ' On ' + str(prediction_score[0][0] * 100)[0:5] + ' % sure there is low risk.'
            color1 = '#1cc78b'
            return render_template("website.html", genders=genders, hypertensions=hypertensions,
                                   heart_diseases=heart_diseases,
                                   ever_marrieds=ever_marrieds, work_types=work_types, Residence_types=Residence_types,
                                   smoking_statuses=smoking_statuses, output=prediction, color=color1)

        if prediction_label == 1:
            # prediction = ' On '+ str(round(prediction_score[0][1] * 100))+' % sure there is high risk.'
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
