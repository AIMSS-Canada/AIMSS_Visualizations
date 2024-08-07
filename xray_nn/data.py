import streamlit as st
import os
import random
import cv2 
import numpy as np

st.title("Data")

def load_image(path):
   files = os.listdir(path)
   file = files[random.randrange(len(files))]
   return path + file

def preprocess(directory):
   file_path = load_image(directory)
   xray_img = cv2.imread(file_path)
   xray_img = xray_img.astype(np.float32)/255
   xray_img = cv2.resize(xray_img, (224, 224))
   xray_img = np.array(xray_img)
   return st.image(xray_img)

st.button('Randomize Images')

col11, col12, col13, col14 = st.columns(4)
col21, col22, col23, col24 = st.columns(4)
col31, col32, col33, col34 = st.columns(4)
col41, col42, col43, col44 = st.columns(4)

with col12:
   st.subheader('Normal')
with col13:
   st.subheader('Covid 19')
with col14:
   st.subheader('Pneumonia')

with col21:
   st.subheader('Training')
   st.write('Data the model will form patterns from.')
with col22:
   preprocess('./xray_nn/data/train/NORMAL/')
with col23:
   preprocess('./xray_nn/data/train/COVID19/')
with col24:
   preprocess('./xray_nn/data/train/PNEUMONIA/')

with col31:
   st.subheader('Validation')
   st.write('Data the program will use to select the best model.')
with col32:
   preprocess('./xray_nn/data/val/NORMAL/')
with col33:
   preprocess('./xray_nn/data/val/COVID19/')
with col34:
   preprocess('./xray_nn/data/val/PNEUMONIA/')

with col41:
   st.subheader('Testing')
   st.write('A holdout set to evaluate final model performance.')
with col42:
   preprocess('./xray_nn/data/test/NORMAL/')
with col43:
   preprocess('./xray_nn/data/test/COVID19/')
with col44:
   preprocess('./xray_nn/data/test/PNEUMONIA/')
