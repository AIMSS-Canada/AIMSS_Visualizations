import streamlit as st
import numpy as np
import cv2
import random
import os
from keras.models import load_model

st.title("Predict")

def load_image(path):
   files = os.listdir(path)
   file = files[random.randrange(len(files))]
   return path + file

def randomize():
    ground_truth = random.choice(['Normal', 'Covid19', 'Pneumonia'])
    if ground_truth == 'Normal':
        st.session_state.image = load_image('./xray_nn/data/test/NORMAL/')
    elif ground_truth == 'Covid19':
        st.session_state.image = load_image('./xray_nn/data/test/COVID19/')
    elif ground_truth == 'Pneumonia':
        st.session_state.image = load_image('./xray_nn/data/test/PNEUMONIA/')
    else:
        st.error('Error with loading image', icon='🚨')

if 'image' not in st.session_state:
    randomize()

col1, col2 = st.columns([2, 1])

with col1:
    e = st.slider(
        'Epochs', 
        min_value = 1,
        max_value = 20,
        value = 10,
    )
    lr = st.select_slider(
        'Learning Rate',
        options = ['1E-3', '1E-4', '1E-5'],
        value = '1E-4',
    )
    st.button('Randomize Image', on_click=randomize)

    model_path = f'./xray_nn/model/lr_{lr}__loss_catcrossent__met_acc/lr_{lr}__e_{e-1}__loss_catcrossent__met_acc.keras'
    model = load_model(model_path, compile=True)

with col2:
    xray_img = cv2.imread(st.session_state.image)
    xray_img = cv2.cvtColor(xray_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    xray_img = xray_img.astype(np.float32) / 255.0
    xray_img = cv2.resize(xray_img, (224, 224))
    st.image(xray_img, use_column_width=True)

    xray_img = np.expand_dims(xray_img, axis=0)

    classes = ['COVID19', 'NORMAL', 'PNEUMONIA'] 

    predictions = model.predict(xray_img) 
    
    st.write(f'Prediction: {classes[np.argmax(predictions)]}')
    st.write(f'Ground Truth: {st.session_state.image.split("/")[-2]}')
