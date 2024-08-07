import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

config = {'displayModeBar': False}
st.title("k-NN")

def euclidean_dist(x, y):
    return np.sqrt((st.session_state['height'] - x) ** 2 + (st.session_state['weight'] - y) ** 2)

df = pd.DataFrame({
    'Sex': ['Male']*10 + ['Female']*10,
    'Height': [
        169.2, 173.29, 174.1, 164.54, 162.70, 174.9, 175.1, 172.56, 176.02, 179.3,  # Male
        163.89, 152.42, 170.32, 160.6, 164.98, 172.2, 161.41, 171.53, 172.2, 163.4,  # Female
    ],
    'Weight': [
        79.6, 79.94, 84.2, 82.8, 83.01, 78.65, 84.21, 84.7, 82.98, 85.21,  # Male
        76.23, 77.87, 79.6, 78.86, 79.1, 79.56, 76.69, 80.4, 81.63, 82.15,  # Female
    ],
})

col1, col2 = st.columns([2.5, 1])

with col2:
    st.number_input(
        'Height', 
        min_value = 0., 
        max_value = 200., 
        step = 0.1, 
        value = 165., 
        key = 'height'
    )
    st.number_input(
        'Weight', 
        min_value = 0., 
        max_value = 200., 
        step = 0.1, 
        value = 80., 
        key = 'weight'
    )
    st.number_input(
        'K',  
        min_value = 0, 
        max_value = 20, 
        value = 5, 
        key = 'k'
    )

    distances = [round(euclidean_dist(df['Height'][i], df['Weight'][i]), 2) for i in range(len(df))]
    k_indices = np.argsort(distances)[:st.session_state['k']]
    k_labels = [df['Sex'][i] for i in k_indices]
    prediction = 'Male' if k_labels.count('Male') > k_labels.count('Female') else 'Female' if k_labels.count('Male') < k_labels.count('Female') else 'Unknown'

    col21, col22 = st.columns(2)
    with col21:
        st.metric('Male', f"{k_labels.count('Male')}/5")
    with col22:
        st.metric('Female', f"{k_labels.count('Female')}/5")

    st.write(f"Predicted Sex: {prediction}")

with col1:    
    fig = go.Figure()
    for i in k_indices:
        fig.add_trace(go.Scatter(
            x = [df['Height'][i]],
            y = [df['Weight'][i]],
            mode = 'markers',
            marker_color = 'yellow',
            marker_size = 20,
            name = 'k-nearest',
            hoverinfo = 'skip',
        ))
    fig.add_trace(go.Scatter(
        x = df[df['Sex']=='Male']['Height'],
        y = df[df['Sex']=='Male']['Weight'],
        customdata = distances[:20],
        mode = 'markers',
        marker_color = '#0068c9',
        marker_symbol = 'square',
        name = 'Male',
        hovertemplate = '''
            Male<br>
            Height: %{x}<br>
            Weight: %{y}<br>
            Euclidean Distance: %{customdata}
            <extra></extra>
        '''
    ))
    fig.add_trace(go.Scatter(
        x = df[df['Sex']=='Female']['Height'],
        y = df[df['Sex']=='Female']['Weight'],
        customdata = distances[10:],
        mode = 'markers',
        marker_color = '#ff2b2b',
        name = 'Female',
        hovertemplate = '''
            Female<br>
            Height: %{x}<br>
            Weight: %{y}<br>
            Euclidean Distance: %{customdata}
            <extra></extra>
        '''
    ))
    fig.add_trace(go.Scatter(
        x = [st.session_state['height']],
        y = [st.session_state['weight']],
        mode = 'markers',
        marker_color = 'black',
        marker_symbol = 'x',
        marker_size = 10,
        name = 'Prediction',
    ))
    fig.update_layout(
        margin = dict(t=0, r=20),
        height = 400,
        showlegend = False,
        xaxis_title = 'Height (cm)',
        yaxis_title = 'Weight (kg)',
    )
    st.plotly_chart(fig, config=config)
