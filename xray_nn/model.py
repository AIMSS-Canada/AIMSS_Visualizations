import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("Model")

lr = st.select_slider(
    'Learning Rate',
    options = ['1E-3', '1E-4', '1E-5'],
    value = '1E-4',
    key = 'lr1',
)

df = pd.read_csv('./xray_nn/model/training.log', sep=',', engine='python')
filtered = df[df['model']==f'lr_{lr}__loss_catcrossent__met_acc']

col1, col2 = st.columns(2)

with col1:
    fig = px.line(
        filtered, 
        x = 'epoch', 
        y = 'loss', 
        title = 'Training and Validation Loss'
    )
    fig.update_traces(
        hovertemplate = 'Train Loss: %{y}<extra></extra>'
    )
    fig.add_trace(go.Scatter(
        x = filtered['epoch'], 
        y = filtered['val_loss'], 
        mode = 'lines',
        showlegend = False,
        name = 'Val Loss',
        hovertemplate = 'Val Loss: %{y}<extra></extra>'
    ))
    fig.update_layout(
        hovermode = 'x unified',
    )
    st.plotly_chart(fig)

with col2:
    fig = px.line(
        filtered, 
        x = 'epoch', 
        y = 'accuracy', 
        title = 'Training and Validation Accuracy'
    )
    fig.update_traces(
        hovertemplate = 'Train Accuracy: %{y}<extra></extra>'
    )
    fig.add_trace(go.Scatter(
        x = filtered['epoch'], 
        y = filtered['val_accuracy'], 
        mode = 'lines',
        showlegend = False,
        name = 'Val Accuracy',
        hovertemplate = 'Val Accuracy: %{y}<extra></extra>'
    ))
    fig.update_layout(
        hovermode = 'x unified',
    )
    st.plotly_chart(fig)