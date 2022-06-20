from email import header
import numpy as np
from pandas import options
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

def app():
    header = st.container()
    dataset = st.container()
    features = st.container()
    model_training = st.container()

    # Loading pickled file.
    model = pickle.load(open('model.pkl', 'rb'))

    def predict(data):
        scale = StandardScaler()
        x_test = scale.fit_transform(data)
        prediction = model.predict(x_test)
        print(prediction)
        return prediction

    with header:

        st.title("Heart Disease Prediction.")


    with dataset:

        st.header("Kindly Input your data")



        age = st.number_input("Please put your age.", 0)
        sex = st.number_input("What is your sexual orientation? 1 for male and 0 for female", 0)
        cp = st.number_input(
            "Input(Chest pain type in number => Value 0: typical angina, Value 1: atypical angina, Value 2: non-anginal pain, Value 3: asymptomatic)",
            0)
        trtbps = st.number_input("Input(resting blood pressure (in mm Hg))", 0)
        chol = st.number_input("Input(cholestoral in mg/dl fetched via BMI sensor)", 0)
        fps = st.number_input("Input(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)", 0)
        restecg = st.number_input("Input( resting electrocardiographic results)", 0)
        thalachh = st.number_input("Input(maximum heart rate achieved)", 0)
        exng = st.number_input("Input(exercise induced angina (1 = yes; 0 = no))", 0)
        caa = st.number_input("Input(number of major vessels (0-3))", 0)

        input_list = ["age", "sex", "cp", "trtbps", "chol", "fps", "restecg", "thalach", "exng", "caa"]
        data = pd.DataFrame(data=[[age, sex, cp, trtbps, chol, fps, restecg, thalachh, exng, caa]],
                            columns=input_list)

        result = ""
        if st.button("Predict"):
            result = predict(data)
            if  result[0] == 1:
                st.error('Yes you have heart disease. Please Consult a doctor')
            else:
                st.success("No you don't have heart disease but take care of your heart.")


