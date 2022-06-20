import streamlit as st
import cv2 
import numpy as np 
from keras.models import load_model

def app():                                                   
    model = load_model('cancer.h5')
    CLASS_NAMES = ['High squamous intra-epithelial lesion', 'Low squamous intra-epithelial lesion', 'Negative for Intraepithelial malignancy', 'Squamous cell carcinoma']
    st.title("Diagnosis of pre-cancerous and cervical cancer lesions")
    st.markdown("Upload an image ")
    plant_image = st.file_uploader("Choose an image...", type="jpg")
    submit = st.button('Predict')
#On predict button click
    if submit:


        if plant_image is not None:

        # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
            st.image(opencv_image, channels="BGR")
            st.write(opencv_image.shape)
        #Resizing the image
            opencv_image = cv2.resize(opencv_image, (180,180))
        #Convert image to 4 Dimension
            opencv_image.shape = (1,180,180,3)
        #Make Prediction
            Y_pred = model.predict(opencv_image)
            result = CLASS_NAMES[np.argmax(Y_pred)]
            st.title(result)
