
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import streamlit as st
from PIL import Image
import statistics


st.title("Automated Road Damage Detection")

def uploading():
    upload = st.file_uploader("Choose a file")  
    if upload is not None:
        st.write(upload.name)
        file_path = os.path.join("results", upload.name)
        with open(file_path, "wb") as user_file:
            user_file.write(upload.getbuffer())

        return file_path, upload.name

 
img_path, name = uploading()


model = YOLO("best.pt")
model.predict(source=img_path, save=True, save_crop=True, project="output", name="inference", exist_ok=True, save_txt=True)


col1, col2 = st.columns(2)

with col1:
    st.image(img_path, caption='Uploaded image', width=350)

with col2:
    st.image('output//inference//' + name, caption='Predicted image', width=350)
