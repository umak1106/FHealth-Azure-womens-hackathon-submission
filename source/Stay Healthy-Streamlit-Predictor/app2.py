import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error


def app():   
    st.title("Mental Health Prediction")

    st.write("""
    #Explore different ML algorithms
    """)
    classifiers = st.sidebar.selectbox("Select Classifier",("KNN","SVC","LogisticRegression","Decision Tree","Random Forest","NaiveBayes"))
    data = pd.read_csv("survey.csv")
    st.image("images/3.png")
    st.write(" Mental health is a state of well-being in which an individual realizes his or her own abilities, can cope with the normal stresses of life")
    st.write("This model helps you predict if you are facing any mental health disorder")
    data.drop(columns=['Timestamp', 'Country', 'state', 'comments'], inplace = True)
    data.drop(data[data['Age'] < 0].index, inplace = True) 
    data.drop(data[data['Age'] > 100].index, inplace = True)
    data['work_interfere'] = data['work_interfere'].fillna('Don\'t know' )
    data['self_employed'] = data['self_employed'].fillna('No')
    data['Gender'].replace(['Male ', 'male', 'M', 'm', 'Male', 'Cis Male',
                     'Man', 'cis male', 'Mail', 'Male-ish', 'Male (CIS)',
                      'Cis Man', 'msle', 'Malr', 'Mal', 'maile', 'Make',], 'Male', inplace = True)

    data['Gender'].replace(['Female ', 'female', 'F', 'f', 'Woman', 'Female',
                     'femail', 'Cis Female', 'cis-female/femme', 'Femake', 'Female (cis)',
                     'woman',], 'Female', inplace = True)

    data["Gender"].replace(['Female (trans)', 'queer/she/they', 'non-binary',
                     'fluid', 'queer', 'Androgyne', 'Trans-female', 'male leaning androgynous',
                      'Agender', 'A little about you', 'Nah', 'All',
                      'ostensibly male, unsure what that really means',
                      'Genderqueer', 'Enby', 'p', 'Neuter', 'something kinda male?',
                      'Guy (-ish) ^_^', 'Trans woman',], 'Other', inplace = True)
    list_col=['Age', 'Gender', 'self_employed', 'family_history', 'treatment',
       'work_interfere', 'no_employees', 'remote_work', 'tech_company',
       'benefits', 'care_options', 'wellness_program', 'seek_help',
       'anonymity', 'leave', 'mental_health_consequence',
       'phys_health_consequence', 'coworkers', 'supervisor',
       'mental_health_interview', 'phys_health_interview',
       'mental_vs_physical', 'obs_consequence']

    for col in list_col: 
        print('{} :{} ' . format(col.upper(),data[col].unique()))

    n_f = data.select_dtypes(include=[np.number]).columns
    c_f = data.select_dtypes(include=[np.object]).columns

    label_encoder = LabelEncoder()
    for col in c_f:
        label_encoder.fit(data[col])
        data[col] = label_encoder.transform(data[col])
    X = data.drop("treatment",axis=1)
    y = data["treatment"]
    def add_parameters_csv(clf_name):
        p = dict()
        if clf_name == "KNN":
            K = st.sidebar.slider("K",1,30)
            p["K"] = K
        elif clf_name == "SVC":
            C = st.sidebar.slider("C",0.01,15.0)
            p["C"] = C
        elif clf_name == "Random Forest":
            max_depth = st.sidebar.slider("max_depth",2,15)
            n_estimators = st.sidebar.slider("n_estimators",1,100)
            p["max_depth"] = max_depth
            p["n_estimators"] = n_estimators
        elif clf_name == "LogisticRegression":
            max_iter = st.sidebar.slider("max_iter",100,300)
            C = st.sidebar.slider("C",1,5)
            p["max_iter"] = max_iter
            p["C"] = C
        elif clf_name == "Decision Tree":
            min_samples_split = st.sidebar.slider("min_samples_split",2,5)
            p["min_samples_split"] = min_samples_split
        else:
            st.write("No Parameters selection")
        return p
    p = add_parameters_csv(classifiers)

    def get_Classifier_csv(clf_name,p):
        if clf_name == "KNN":
            clf = KNeighborsClassifier(n_neighbors=p["K"])
        elif clf_name == "SVC":
            clf = SVC(C=p["C"])
        elif clf_name == "Random Forest":
            clf = RandomForestClassifier(n_estimators=p["n_estimators"],max_depth=p["max_depth"],random_state=1200)
        elif clf_name == "LogisticRegression":
            clf = LogisticRegression(C=p["C"],max_iter=p["max_iter"])
        elif clf_name == "NaiveBayes":
            clf = GaussianNB()
        else:
            clf = DecisionTreeClassifier(min_samples_split=p["min_samples_split"])
        return clf
    clf = get_Classifier_csv(classifiers,p)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1200)
    clf.fit(X_train,y_train)
    y_pred_test = clf.predict(X_test)
    st.write(f"classifier Used={classifiers}")
    acc = accuracy_score(y_test,y_pred_test)
    st.write(f"accuracy score={acc}")
    st.write("Your Age ")
    age = st.number_input("age",0,100)
    st.write("Gender Female:0 , Male : 1 , Other:2")
    Gender = st.number_input("Gender",0,2)
    st.write("self_employed No:0 , Yes:1")
    self_employed = st.number_input("self_employed",0,1)
    st.write(" Do you have a family history of mental illness?")
    st.write("family_history N0:0 , Yes:1")
    family_history = st.number_input("family_history",0,1)
    st.write("If you have a mental health condition, do you feel that it interferes with your work?")
    st.write("work_interfere Often:0 , Rarely:1 , Never:2 , Sometimes:3 , Don't know:4")
    work_interfere = st.number_input("work_interfere",0,4)
    st.write("Number of employees")
    no_employees = st.number_input("no_employees",0,1000)
    st.write("Do you work remotely (outside of an office) at least 50% of the time?")
    st.write("remote_work No:0 , Yes:1")
    remote_work = st.number_input("remote_work",0,2)
    st.write("Is your employer primarily a tech company/organization?")
    st.write("tech_company Yes:0 , No:1")
    tech_company = st.number_input("tech_company",0,1)
    st.write("Does your employer provide mental health benefits?")
    st.write("benefits Yes:0 , Don't know:1 ,No:2")
    benefits = st.number_input("benefits",0,2)
    st.write("Do you know the options for mental health care your employer provides?")
    st.write("care_options NotSure:0,No:1,Yes:2")
    care_options = st.number_input("care_options",0,2)
    st.write("Has your employer ever discussed mental health as part of an employee wellness program?")
    st.write("wellness_program No:0,Don't Know:1,Yes:2")
    wellness_program = st.number_input("wellness_program",0,2)
    st.write("Does your employer provide resources to learn more about mental health issues and how to seek help?")
    st.write("seek_help Yes:0,Don't know:1,Yes:2")
    seek_help = st.number_input("seek_help",0,2)
    st.write(" Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources?")
    st.write("anonymity Yes:0 , Don't know:1 , N0:2")
    anonymity= st.number_input("anonymity",0,2)
    st.write("How easy is it for you to take medical leave for a mental health condition?")
    st.write("leave Somewhat easy:0 , Don't know:1 ,Somewhat difficult:2 ,Very difficult:3 ,Very easy:4")
    leave= st.number_input("leave",0,4)
    st.write("Do you think that discussing a mental health issue with your employer would have negative consequences?")
    st.write("mental_health_consequence No:0 ,Maybe:1 , Yes:2 ")
    mental_health_consequence = st.number_input("mental_health_consequence",0,2)
    st.write("Do you think that discussing a physical health issue with your employer would have negative consequences?")
    st.write("phys_health_consequence No:0 , Yes:1 , Maybe:2")
    phys_health_consequence= st.number_input("phys_health_consequence",0,2)
    st.write("Would you be willing to discuss a mental health issue with your coworkers?")
    st.write("coworkwers Some of them:0 , No:1 , Yes:2")
    coworkers= st.number_input("coworkers",0,2)
    st.write("Would you be willing to discuss a mental health issue with your direct supervisor(s)?")
    st.write("supervisor Yes:0 , No:1 , Some of them:2")
    supervisor= st.number_input("supervisor",0,2)
    st.write("Would you bring up a mental health issue with a potential employer in an interview?")
    st.write("mental_health_interview No:0 , Yes:1 , Maybe:2")
    mental_health_interview= st.number_input("mental_health_interview",0,2)
    st.write("Would you bring up a physical health issue with a potential employer in an interview?")
    st.write("phys_health_interview Maybe:0 , No:1 , Yes:2")
    phys_health_interview= st.number_input("phys_health_interview",0,2)
    st.write("Do you feel that your employer takes mental health as seriously as physical health?")
    st.write("mental_vs_physical Yes:0 , Don't know:1 , No:2")
    mental_vs_physical= st.number_input("mental_vs_physical",0,2)
    st.write("Have you heard of or observed negative consequences for coworkers with mental health conditions in your workplace?")
    st.write("obs_consequence No:0 , Yes:1")
    obs_consequence= st.number_input("obs_consequence",0,1)

    data1={'Age':age, 'Gender':Gender, 'self_employed':self_employed, 'family_history':family_history,
       'work_interfere':work_interfere, 'no_employees':no_employees, 'remote_work':remote_work, 'tech_company':tech_company,
       'benefits':benefits, 'care_options':care_options, 'wellness_program':wellness_program, 'seek_help':seek_help,
       'anonymity':anonymity, 'leave':leave, 'mental_health_consequence':mental_health_consequence,
       'phys_health_consequence':phys_health_consequence, 'coworkers':coworkers, 'supervisor':supervisor,
       'mental_health_interview':mental_health_interview, 'phys_health_interview':phys_health_interview,
       'mental_vs_physical':mental_vs_physical, 'obs_consequence':obs_consequence}
    df2 = pd.DataFrame(data1,index=["Name"])
    y_pred_test1 = clf.predict(df2)

    if(y_pred_test1==0):
        st.write("You Need a Treatment")

    else:
        st.write("You do not Need a Treatment")

