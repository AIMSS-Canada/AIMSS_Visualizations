import streamlit as st
import pandas as pd
from math import floor, ceil
import plotly.graph_objects as go

st.set_page_config(layout="centered")
config = {'displayModeBar': False}
st.title("Linear Regression")

if 'lin_m' not in st.session_state:
    st.session_state['lin_m'] = 6.0
if 'lin_b' not in st.session_state:
    st.session_state['lin_b'] = 79.0
if 'lin_x' not in st.session_state:
    st.session_state['lin_x'] = 9.5

df = pd.DataFrame({
    'Age': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    'Height': [110.3, 116.0, 121.7, 127.3, 132.6, 137.8, 143.1, 149.1, 156.0, 163.2, 169.0],
})

with st.expander('Help'):
    st.write('''
    The points illustrate the median height of boys from ages 5 to 15, as collected by WHO. If you were to plot the line all the way to age 19, you would see
    that the line is not truly linear since growth typically slows into adulthood. Linear regression would not be able to capture the plateau in growth.

    The goal of linear regrssion is to find the optimal equation for a line to minimize loss. In this case, you can play with the line's equation and see how it 
    affects the prediction. A line that is not representative of the data will output a prediction that is not representative of the data. 
            
    **Hint:** Slope affects the angle of the line and y-intercept affects the vertical position of the line.
    ''')
    
    st.divider()

    st.write('''
    **Note:** Linear regression is biased in that the model always assumes a linear relation. In this context, the range of ages should be limited or else 
    the model will increase height infinitely. Try inputting a very low or high age, and the regression will return an unlikely or impossible height.
    ''')

col1, col2 = st.columns(2)

with col1:
    st.slider(
        'Slope', 
        min_value = 0., 
        max_value = 10., 
        step = 0.1, 
        value = 6., 
        key = 'lin_m',
    )

with col2:
    st.slider(
        'Y-Intercept', 
        min_value = 50., 
        max_value = 90., 
        step = 0.1, 
        value = 79., 
        key = 'lin_b',
    )

col1, col2 = st.columns([2.5, 1])

with col2:
    st.number_input(
        'Age', 
        min_value = -1000.0, 
        max_value = 1000.0, 
        step = 0.5, 
        value = 9.5, 
        key = 'lin_x'
    )

    prediction = st.session_state['lin_m'] * st.session_state['lin_x'] + st.session_state['lin_b']
    st.write(f"Predicted Height: {round(prediction, 2)} cm")

with col1:
    fig = go.Figure()
    line_range = range(floor(min(5, st.session_state['lin_x'])-1), ceil(max(16, st.session_state['lin_x']))+1)
    fig.add_trace(go.Scatter(
        x = list(line_range),
        y = [st.session_state['lin_m'] * i + st.session_state['lin_b'] for i in line_range],
        mode = 'lines',
        marker_color = '#bfc5d3',
        hoverinfo = 'skip',
    ))
    fig.add_trace(go.Scatter(
        x = df['Age'],
        y = df['Height'],
        mode = 'markers',
        marker_color = '#0068c9',
        name = 'GroundTruth',
    ))
    if 'lin_x' in st.session_state:
        fig.add_trace(go.Scatter(
            x = [st.session_state['lin_x']],
            y = [prediction],
            mode = 'markers',
            marker_color = 'black',
            marker_size = 10,
            marker_symbol = 'x',
            name = 'Prediction',
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