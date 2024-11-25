import streamlit as st
import numpy as np
import cv2
import random
import os
from keras.models import load_model

st.set_page_config(layout="centered")

st.title("Predict")

def load_image(path):
   files = os.listdir(path)
   file = files[random.randrange(len(files))]
   return path + file

def randomize(key):
    ground_truth = random.choice(['Normal', 'Covid19', 'Pneumonia'])
    if ground_truth == 'Normal':
        st.session_state[key] = load_image('./xray_nn/data/test/NORMAL/')
    elif ground_truth == 'Covid19':
        st.session_state[key] = load_image('./xray_nn/data/test/COVID19/')
    elif ground_truth == 'Pneumonia':
        st.session_state[key] = load_image('./xray_nn/data/test/PNEUMONIA/')
    else:
        st.error('Error with loading image', icon='🚨')

def randomize_all():
    randomize('image1')
    randomize('image2')
    randomize('image3')

def plot_image(img):
    xray_img = cv2.imread(img)
    xray_img = cv2.cvtColor(xray_img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    xray_img = xray_img.astype(np.float32) / 255.0
    xray_img = cv2.resize(xray_img, (224, 224))
    st.image(xray_img, use_container_width=True)
    return xray_img

if 'model' not in st.session_state:
    e = 20
    lr = '1E-4'
    model_path = f'./xray_nn/model/lr_{lr}__loss_catcrossent__met_acc/lr_{lr}__e_{e-1}__loss_catcrossent__met_acc.keras'
    st.session_state['model'] = load_model(model_path, compile=True)

if 'image1' not in st.session_state:
    randomize('image1')
if 'image2' not in st.session_state:
    randomize('image2')
if 'image3' not in st.session_state:
    randomize('image3')

classes = ['COVID19', 'NORMAL', 'PNEUMONIA']

st.button('Randomize Images', on_click=randomize_all)

col1, col2, col3 = st.columns(3)
with col1:
    xray_img1 = plot_image(st.session_state['image1'])
    xray_img1 = np.expand_dims(xray_img1, axis=0)

    predictions = st.session_state['model'].predict(xray_img1) 
    
    st.write(f'Prediction: {classes[np.argmax(predictions)]}')
    st.write(f'Ground Truth: {st.session_state.image1.split("/")[-2]}')

with col2:
    xray_img2 = plot_image(st.session_state['image2'])
    xray_img2 = np.expand_dims(xray_img2, axis=0)

    predictions = st.session_state['model'].predict(xray_img2) 
    
    st.write(f'Prediction: {classes[np.argmax(predictions)]}')
    st.write(f'Ground Truth: {st.session_state.image2.split("/")[-2]}')

with col3:
    xray_img3 = plot_image(st.session_state['image3'])
    xray_img3 = np.expand_dims(xray_img3, axis=0)

    predictions = st.session_state['model'].predict(xray_img3) 
    
    st.write(f'Prediction: {classes[np.argmax(predictions)]}')
    st.write(f'Ground Truth: {st.session_state.image3.split("/")[-2]}')
    

     

    