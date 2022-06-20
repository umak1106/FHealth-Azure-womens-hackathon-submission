import streamlit as st
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def app():
    classificationModel = st.sidebar.selectbox("Model", ["Random Forest Classifier", "GaussianNB", "K Nearest Neighbours", "Decision Tree Classifier"])
    texture = st.sidebar.slider('Texture', 9.0, 40.0, 12.0)
    perimeter = st.sidebar.slider('Perimeter', 40.0, 190.0, 100.0)
    smoothness = st.sidebar.slider('Smoothness', 0.01, 0.18, 0.10)
    compactness = st.sidebar.slider('Compactness', 0.005, 0.400, 0.200)


    st.title('Breast Cancer  Prediction')

    data = pd.read_csv('dataset.csv', header=0)

    ##Cleaning dataset
    data.drop('id', axis=1, inplace = True)
    data.drop("Unnamed: 32",axis=1,inplace=True)

    features_mean = list(data.columns[1:11])
    features_se = list(data.columns[11:20])
    features_worst = list(data.columns[20:31])


    ## Mapping malignant to 1 and benign to 0
    data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})


    ## Preparing data for the model
    prediction_var = ['texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean']
    train, test = train_test_split(data, test_size = 0.3)
    train_x = train[prediction_var]
    train_y = train.diagnosis

    test_x = test[prediction_var]
    test_y = test.diagnosis

    ## Selecting model to use
    if classificationModel == "Random Forest Classifier":
        model = RandomForestClassifier(n_estimators=100)

    if classificationModel == "GaussianNB":
        model = GaussianNB()

    if classificationModel == "K Nearest Neighbours":
        model = KNeighborsClassifier()

    if classificationModel == "Decision Tree Classifier":
        model = DecisionTreeClassifier()


    model.fit(train_x, train_y)
    prediction = model.predict(test_x)

    accuracy = "Accuracy =", metrics.accuracy_score(prediction, test_y)
    accuracy

    prediction2 = model.predict([[texture, perimeter, smoothness, compactness]])


    st.header('Prediction')


    if (prediction2 == 0):
        st.success("Benign")

    else:
        st.success("Malignant")
