import streamlit as st
import pandas as pd
import numpy as np
from math import floor, ceil
import plotly.graph_objects as go

st.set_page_config(layout="centered")
config = {'displayModeBar': False}
st.title("Outliers")

if 'lin_m' not in st.session_state:
    st.session_state['lin_m'] = 6.0
if 'lin_b' not in st.session_state:
    st.session_state['lin_b'] = 79.0
if 'lin_x' not in st.session_state:
    st.session_state['lin_x'] = 9.5

df = pd.DataFrame({
    'Age': [5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15],
    'Height': [111.34325594769082, 109.7663812408017, 109.86390294108935, 109.54623979990068, 109.60342836778602, 117.77389643224386, 115.53643085179522, 116.394580564921, 115.94178696222664, 113.96546506186014, 121.04389581980938, 120.59701156173985, 121.28487664057832, 122.61670364461098, 120.48117993291004, 127.2164151806974, 125.1597987111308, 128.7098296596255, 129.0326845912423, 127.26759058215151, 131.96104460129843, 131.35640356275624, 134.8064013018051, 130.4969568265708, 133.70895876948873, 139.33396646893817, 138.25528672816972, 136.52673880930396, 139.0039271883908, 138.21972310528193, 144.23333514955033, 143.49560708706355, 142.50896694387575, 145.7817272077502, 144.51669903794195, 149.49325528385367, 150.256139872925, 145.48520452092097, 148.00693809044597, 149.87935211342193, 154.75061775764578, 158.56790138978883, 156.16713764598123, 155.4374410714955, 155.99559513120278, 163.79537422441987, 163.77246405878148, 163.63188818001842, 164.80624769325215, 163.11101478249674, 170.48148139553098, 168.26380002997453, 167.35953099992966, 169.56322641112698, 172.63029735015198],
})

# Number of additional data points per age
num_points_per_age = 5

# Create new data points by adding random noise
crowded_data = []

for age, height in zip(df['Age'], df['Height']):
    for _ in range(num_points_per_age):
        noisy_height = height + np.random.normal(0, 1.5)  # Adding random noise
        crowded_data.append({'Age': age, 'Height': noisy_height})

# Create a new DataFrame from the crowded data
crowded_df = pd.DataFrame(crowded_data)

# Display the new DataFrame
print(crowded_df['Age'].tolist())
print(crowded_df['Height'].tolist())

col1, col2 = st.columns(2)

with col1:
    st.slider(
        'Slope', 
        min_value = 0., 
        max_value = 10., 
        step = 0.1,
        key = 'lin_m',
    )

with col2:
    st.slider(
        'Y-Intercept', 
        min_value = 50., 
        max_value = 90., 
        step = 0.1, 
        key = 'lin_b',
    )

col1, col2 = st.columns([2.5, 1])

with col2:
    st.number_input(
        'Age', 
        min_value = -1000.0, 
        max_value = 1000.0, 
        step = 0.5, 
        key = 'lin_x'
    )

    prediction = st.session_state['lin_m'] * st.session_state['lin_x'] + st.session_state['lin_b']
    st.write(f"Predicted Height: {round(prediction, 2)} cm")

with col1:
    fig = go.Figure()
    line_range = range(floor(min(5, st.session_state['lin_x'])-1), ceil(max(16, st.session_state['lin_x']))+1)
    # fig.add_trace(go.Scatter(
    #     x = list(line_range),
    #     y = [st.session_state['lin_m'] * i + st.session_state['lin_b'] for i in line_range],
    #     mode = 'lines',
    #     marker_color = '#bfc5d3',
    #     hoverinfo = 'skip',
    # ))
    fig.add_trace(go.Scatter(
        x = df['Age'],
        y = df['Height'],
        mode = 'markers',
        marker_color = '#0068c9',
        name = 'GroundTruth',
    ))
    fig.update_layout(
        margin = dict(t=50, r=0),
        xaxis_title = 'Age (years)',
        yaxis_title = 'Height (cm)',
        height = 400,
        showlegend = False,
        title = f"Height = {st.session_state['lin_m']} * Age + {st.session_state['lin_b']}",
    )
    st.plotly_chart(fig, config=config)