import streamlit as st
import numpy as np
import os
from tensorflow.keras.preprocessing import image

class FilesUpload(object):

    def __init__(self):
        self.fileTypes = ["png", "jpg", "jpeg"]

    def run(self):
        
        image_shape = (130, 130, 3)
        st.set_option('deprecation.showfileUploaderEncoding', False)
        img_file = st.file_uploader("Upload file", type = self.fileTypes)
        show_img_file = st.empty()
        
        if not img_file:
            show_img_file.info("Please upload a file of type: " + ", ".join(["png", "jpg", "jpeg"]))
  
        if img_file is not None:
            
            st.image(img_file, width = 130) #use_column_width=True
            file_details = {"File Name" : img_file.name, "File Type" : img_file.type, "File Size" : img_file.size}
            st.write(file_details)
            
            with open(os.path.join("./temp/", img_file.name), 'wb') as f:
                f.write(img_file.getbuffer())
                file_path = "./temp/" + img_file.name

                           
            
            img_file = image.load_img(file_path, grayscale = False, color_mode = 'rgb', target_size = image_shape, interpolation = 'nearest')
            img_file = image.img_to_array(img_file)
            img_file = np.expand_dims(img_file, axis = 0)
            img_file = img_file/255 #Normalizing the Image

            st.text(img_file.shape)
            st.text("image = {}, Width = {}, Height = {}, color = {}".format(img_file.shape[0],
                    img_file.shape[1], img_file.shape[2], img_file.shape[3]))
            
            os.remove(file_path)
            
            return img_file
                
        else:
            st.write("Incorrect file or file extension")
