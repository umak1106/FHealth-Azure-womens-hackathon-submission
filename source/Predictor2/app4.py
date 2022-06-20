import streamlit as st
import cv2 
import numpy as np 
from keras.models import load_model

def app():                                                   
    model = load_model('cancer.h5')
    CLASS_NAMES = ['High squamous intra-epithelial lesion', 'Low squamous intra-epithelial lesion', 'Negative for Intraepithelial malignancy', 'Squamous cell carcinoma']
    st.title(" pre-cancerous lesions prediction")
    st.markdown("Upload an image ")
    cancer_image = st.file_uploader("Choose an image...", type="jpg")
    submit = st.button('Predict')
    if submit:
        if cancer_image is not None:       
            file_bytes = np.asarray(bytearray(cancer_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)      
            st.image(opencv_image, channels="BGR")
            st.write(opencv_image.shape)       
            opencv_image = cv2.resize(opencv_image, (180,180))      
            opencv_image.shape = (1,180,180,3)      
            Y_pred = model.predict(opencv_image)
            result = CLASS_NAMES[np.argmax(Y_pred)]
            st.title(result)