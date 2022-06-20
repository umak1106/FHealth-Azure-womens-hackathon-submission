import pandas as pd
import streamlit as st
import numpy as np
from joblib import load

def app():
    st.title("PCOS Prediction")
    option = st.sidebar.selectbox(
        'Select an option',
        ('Basic', 'Advanced'))

    model0 = load('RandomForestModel.joblib')
    model1 = load('RandomForestModel1.joblib')

    def displayform():
        if option == 'Basic':
            simple_form = st.form("simple form", clear_on_submit=False)
            age = simple_form.text_input("Age")
            weight = simple_form.text_input("Weight")
            height = simple_form.text_input("Height")
            bmi = simple_form.text_input("BMI")
            cycle_len = simple_form.text_input("Cycle Length (days)")
            weight_gain = simple_form.radio("Weight gain(Y/N)", ["Yes", "No"])
            hair_growth = simple_form.radio("Hair growth(Y/N)", ["Yes", "No"])
            skin_darkening = simple_form.radio("Skin darkening (Y/N)", ["Yes", "No"])
            hair_loss = simple_form.radio("Hair loss(Y/N)", ["Yes", "No"])
            pimples = simple_form.radio("Pimples(Y/N)", ["Yes", "No"])
            fast_food = simple_form.radio("Fast food (Y/N)", ["Yes", "No"])
            reg_exercise = simple_form.radio("Reg.Exercise(Y/N)", ["Yes", "No"])
            pulse_rate = simple_form.text_input("Pulse Rate")
            rr = simple_form.text_input("Respiratory rate (breaths/min)")
            pregnant = simple_form.radio("Pregnant", ["Yes", "No"])
            submitted = simple_form.form_submit_button("Submit")
            if submitted:
                displayresult(0, [age, weight, height, bmi, cycle_len, weight_gain,
                                  hair_growth, skin_darkening, hair_loss, pimples, fast_food, reg_exercise, pulse_rate,
                                  rr, pregnant])
        else:
            advanced_form = st.form("advanced form", clear_on_submit=False)
            age = advanced_form.text_input("Age")
            weight = advanced_form.text_input("Weight")
            height = advanced_form.text_input("Height")
            bmi = advanced_form.text_input("BMI")
            pulse_rate = advanced_form.text_input("Pulse Rate")
            rr = advanced_form.text_input("Respiratory rate (breaths/min)")
            hb = advanced_form.text_input("Hb(g/dl")
            cycle_len = advanced_form.text_input("Cycle Length (days)")
            pregnant = advanced_form.radio("Pregnant", ["Yes", "No"])
            beta_hcg = advanced_form.text_input("beta-HCG(mIU/mL")
            fsh = advanced_form.text_input("FSH(mIU/mL")
            lh = advanced_form.text_input("LH(mIU/mL")
            fsh_lh = advanced_form.text_input("FSH/LH")
            hip = advanced_form.text_input("Hip(inch)")
            waist = advanced_form.text_input("Waist(inch)")
            waist_hip_ratio = advanced_form.text_input("Waist:Hip Ratio")
            tsh = advanced_form.text_input("TSH (mIU/L)")
            prl = advanced_form.text_input("PRL(ng/mL")
            vit_d3 = advanced_form.text_input("Vit D3 (ng/mL)")
            prg = advanced_form.text_input("PRG(ng/mL)")
            rbs = advanced_form.text_input("RBS(mg/dl)")
            weight_gain = advanced_form.radio("Weight gain(Y/N)", ["Yes", "No"])
            hair_growth = advanced_form.radio("Hair growth(Y/N)", ["Yes", "No"])
            skin_darkening = advanced_form.radio("Skin darkening (Y/N)", ["Yes", "No"])
            hair_loss = advanced_form.radio("Hair loss(Y/N)", ["Yes", "No"])
            pimples = advanced_form.radio("Pimples(Y/N)", ["Yes", "No"])
            fast_food = advanced_form.radio("Fast food (Y/N)", ["Yes", "No"])
            reg_exercise = advanced_form.radio("Reg.Exercise(Y/N)", ["Yes", "No"])
            bp_systolic = advanced_form.text_input("BP Systolic (mmHg)")
            bp_diastolic = advanced_form.text_input("BP Diastolic (mmHg)")
            follicle_no_l = advanced_form.text_input("Follicle No. (L)")
            follicle_no_r = advanced_form.text_input("Follicle No. (R)")
            avg_f_size_l = advanced_form.text_input("Average Follicle size (L) (mm)")
            avg_f_size_r = advanced_form.text_input("Average Follicle size (R) (mm)")
            endometrium = advanced_form.text_input("Endometrium (mm)")

            submitted = advanced_form.form_submit_button("Submit")
            if submitted:
                displayresult(1, [age, weight, height, bmi, pulse_rate, rr, hb, cycle_len, pregnant, beta_hcg, fsh, lh,
                                  fsh_lh, hip, waist, waist_hip_ratio,
                                  tsh, prl, vit_d3, prg, rbs, weight_gain, hair_growth, skin_darkening, hair_loss,
                                  pimples, fast_food, reg_exercise,
                                  bp_systolic, bp_diastolic, follicle_no_l, follicle_no_r, avg_f_size_l, avg_f_size_r,
                                  endometrium])

    def displayresult(n, data):
        for i in range(len(data)):
            if data[i] == "Yes":
                data[i] = 1
            elif data[i] == "No":
                data[i] = 0
            else:
                data[i] = float(data[i])
        if n == 0:
            pred = model0.predict(np.array([data]))
        else:
            pred = model1.predict(np.array([data]))
        if pred[0] == 1.0:

            st.error("You have PCOS")
        else:

            st.success("You don't have PCOS")

    displayform()