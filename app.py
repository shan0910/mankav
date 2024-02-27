#import relevant libraries for flask, html rendering and loading the ML model

from flask import Flask,request, url_for, redirect, render_template
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


#loading the SVM model and the preprocessor
model = pickle.load(open("diabetes.pkl", "rb"))
std = pickle.load(open('std.pkl','rb'))

app = Flask(__name__)

#Index.html will be returned for the input
@app.route('/')
def home():
    return render_template("index.html")

############################################################################
#predict function, POST method to take in inputs
@app.route('/predict',methods=['POST','GET'])
def predict():

    #take inputs for all the attributes through the HTML form
    pregnancies = request.form['pregnancies']
    glucose = request.form['glucose']
    bloodpressure = request.form['bloodpressure']
    skinthickness = request.form['skinthickness']
    insulin = request.form['insulin']
    bmi = request.form['bmi']
    diabetespedigreefunction = request.form['diabetespedigreefunction']
    age = request.form['age']
 

    #form a dataframe with the inpus and run the preprocessor as used in the training 
    row_df = pd.DataFrame([pd.Series([pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age])])
    row_df =  pd.DataFrame(std.transform(row_df))
	
    print(row_df)

    #predict the probability and return the probability of being a diabetic
    prediction=model.predict_proba(row_df)
    output='{0:.{1}f}'.format(prediction[0][1], 2)
    output_print = str(float(output)*100)+'%'
    
    if float(output)>0.5:
        return render_template('result_diabetes.html',pred=f'You having chances of getting diabetic.\nProbability of you being diabetic is {output_print}.\n Please Consult to the doctor & try to lower your blood sugar level naturally.')
    else:
        return render_template('result_diabetes.html',pred=f'Congratulations, you are safe but do Exercise regularly.\n Probability of you being a diabetic is {output_print}.')

################################################################################
    
    # loading and reading the dataset

heart = pd.read_csv("heart_cleveland_upload.csv")

# creating a copy of dataset so that will not affect our original dataset.
heart_df = heart.copy()

# Renaming some of the columns 
heart_df = heart_df.rename(columns={'condition':'target'})
print(heart_df.head())

# model building 

#fixing our data in x and y. Here y contains target data and X contains rest all the features.
x= heart_df.drop(columns= 'target')
y= heart_df.target

# splitting our dataset into training and testing for this we will use train_test_split library.
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=42)

#feature scaling
scaler= StandardScaler()
x_train_scaler= scaler.fit_transform(x_train)
x_test_scaler= scaler.fit_transform(x_test)

# creating K-Nearest-Neighbor classifier
model1=RandomForestClassifier(n_estimators=20)
model1.fit(x_train_scaler, y_train)
y_pred= model1.predict(x_test_scaler)
p = model1.score(x_test_scaler,y_test)
print(p)

print('Classification Report\n', classification_report(y_test, y_pred))
print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Creating a pickle file for the classifier
filename = 'heart-disease-prediction-knn-model.pkl'
pickle.dump(model1, open(filename, 'wb'))


# Load the Random Forest CLassifier model
filename = 'heart-disease-prediction-knn-model.pkl'
model1 = pickle.load(open(filename, 'rb'))

@app.route('/heartReport', methods=['GET','POST'])
def heartReport():
     if request.method == 'POST':
     
        age = request.form['age']
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = request.form['trestbps']
        chol = request.form['chol']
        fbs = request.form.get('fbs')
        restecg = request.form['restecg']
        thalach = request.form['thalach']
        exang = request.form.get('exang')
        oldpeak = request.form['oldpeak']
        slope = request.form.get('slope')
        ca = request.form['ca']
        thal = request.form.get('thal')
        
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        my_prediction = model1.predict(data)
        
        return render_template('result_heart.html', prediction=my_prediction)
     


     #######################################################################################


# loading and reading the dataset
# liver = pd.read_csv("indian_liver_patient.csv")

# creating a copy of dataset so that will not affect our original dataset.
# liver_df = liver.copy()

# Renaming some of the columns 
# liver_df = liver_df.rename(columns={'condition':'target'})
# print(liver_df.head())

# # model building 

# #fixing our data in x and y. Here y contains target data and X contains rest all the features.
# x= liver_df.drop(columns= 'target')
# y= liver_df.target

# # splitting our dataset into training and testing for this we will use train_test_split library.
# x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=42)

# #feature scaling
# scaler= StandardScaler()
# x_train_scaler= scaler.fit_transform(x_train)
# x_test_scaler= scaler.fit_transform(x_test)

# # creating K-Nearest-Neighbor classifier
# model2=RandomForestClassifier(n_estimators=20)
# model2.fit(x_train_scaler, y_train)
# y_pred= model2.predict(x_test_scaler)
# p = model2.score(x_test_scaler,y_test)
# print(p)

# print('Classification Report\n', classification_report(y_test, y_pred))
# print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))

# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# Creating a pickle file for the classifier
# filename = 'liver_model(1).pkl'
# pickle.dump(model2, open(filename, 'wb'))

# # Load the Random Forest CLassifier model
# filename = 'liver_model(1).pkl'
# model2 = pickle.load(open(filename, 'rb'))

# @app.route('/liverReport', methods=['GET','POST'])
# def liverReport():
#      if request.method == 'POST':

#         Age = request.form['Age']
#         Gender = request.form.get('Gender')
#         Total_Bilirubin = request.form.get('Total_Bilirubin')
#         Direct_Bilirubin = request.form['Direct_Bilirubin']
#         Alkaline_Phosphotase = request.form['Alkaline_Phosphotase']
#         Alamine_Aminotransferase = request.form.get('Alamine_Aminotransferase')
#         Aspartate_Aminotransferase = request.form['Aspartate_Aminotransferase']
#         Total_Protiens = request.form['Total_Protiens']
#         # Albumin = request.form.get('Albumin')
#         # Albumin_and_Globulin_Ratio = request.form['Albumin_and_Globulin_Ratio']
        
        
#         data = np.array([[Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens]])
#         my_prediction2 = model2.predict(data)
        
#         return render_template('result_liver.html', prediction_Liver=my_prediction2)


def ValuePred(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==10):
        loaded_model = joblib.load("liver.pkl")
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/liverReport', methods=["POST"])
def liverReport():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 10:
            result = ValuePred(to_predict_list, 10)

    if int(result) == 1:
        prediction = "Patient has a high risk of Liver Disease, please consult your doctor immediately."
    else:
        prediction = "Patient has a low risk of Liver Disease."
    return render_template('result_liver.html', prediction_text=prediction)




############################################################################################################

def ValuePred(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==5):
        loaded_model = joblib.load("kidney.pkl")
        result = loaded_model.predict(to_predict)
    return result[0]

@app.route('/kidneyReport', methods=["POST"])
def kidneyReport():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 5:
            result = ValuePred(to_predict_list, 5)

    if(int(result) == 1):
        prediction = "Patient has a high risk of Kidney Disease, please consult your doctor immediately"
    else:
        prediction = "Patient has a low risk of Kidney Disease"
    return render_template("result_kidney.html", prediction_text=prediction)



if __name__ == '__main__':
    app.run(debug=True)