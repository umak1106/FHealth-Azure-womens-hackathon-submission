import streamlit as st
import cv2 
import numpy as np 
from keras.models import load_model

def app():     
    model = load_model('covid.h5')
    CLASS_NAMES = ['negative', 'positive']
    st.title("Covid 19 Classification")

    st.image('images/7.png')

    plant_image = st.file_uploader("Choose an image...", type="jpg")
    submit = st.button('Predict')
    if submit:
        if plant_image is not None:
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            st.image(opencv_image, channels="BGR")
            opencv_image = cv2.resize(opencv_image, (48,48))
            opencv_image.shape = (1,48,48,3)
            pred = model.predict(opencv_image)
            print(np.argmax(pred))
            st.title(str(CLASS_NAMES[np.argmax(pred)]))
