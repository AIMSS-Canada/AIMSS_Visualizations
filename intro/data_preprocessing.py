import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="centered")
config = {'displayModeBar': False}
st.title("Data Preprocessing")

# ----------------------------------
st.subheader('Imputation')
# ----------------------------------

def color_nones_male(x):
    color = 'background-color: yellow'
    df = pd.DataFrame('', index=x.index, columns=x.columns)
    df.iloc[1, 1] = color
    df.iloc[4, 1] = color
    df.iloc[7, 1] = color
    df.iloc[2, 2] = color
    df.iloc[6, 2] = color
    return df

def color_nones_female(x):
    color = 'background-color: yellow'
    df = pd.DataFrame('', index=x.index, columns=x.columns)
    df.iloc[5, 1] = color
    df.iloc[8, 1] = color
    df.iloc[2, 2] = color
    df.iloc[4, 2] = color
    df.iloc[9, 2] = color
    return df

df = pd.DataFrame({
    'Sex': ['Male']*10 + ['Female']*10,
    'Height': [
        169.2, None, 174.1, 164.54, None, 174.9, 175.1, None, 176.02, 179.3,  # Male
        163.89, 152.42, 170.32, 160.6, 164.98, None, 161.41, 171.53, None, 163.4,  # Female
    ],
    'Weight': [
        79.6, 79.94, None, 82.8, 83.01, 78.65, None, 84.7, 82.98, 85.21,  # Male
        76.23, 77.87, None, 78.86, None, 79.56, 76.69, 80.4, 81.63, None,  # Female
    ],
})

col1, col2, col3 = st.columns([1, 2, 2])

with col1:
    st.button('Revert', use_container_width=True, key='revert')
    st.button('Overall Mean', use_container_width=True, key='overall_mean')
    st.button('Overall Median', use_container_width=True, key='overall_median')
    st.button('Overall Mode', use_container_width=True, key='overall_mode')
    st.button('Class Mean', use_container_width=True, key='class_mean')
    st.button('Class Median', use_container_width=True, key='class_median')
    st.button('Class Mode', use_container_width=True, key='class_mode')

    if st.session_state['revert']:
        df = df

    if st.session_state['overall_mean']:
        df['Height'] = df['Height'].fillna(round(df['Height'].mean(), 2))
        df['Weight'] = df['Weight'].fillna(round(df['Weight'].mean(), 2))

    if st.session_state['overall_median']:
        df['Height'] = df['Height'].fillna(round(df['Height'].median(), 2))
        df['Weight'] = df['Weight'].fillna(round(df['Weight'].median(), 2))

    if st.session_state['overall_mode']:
        df['Height'] = df['Height'].fillna(round(df['Height'].mode().mean(), 2))
        df['Weight'] = df['Weight'].fillna(round(df['Weight'].mode().mean(), 2))

    if st.session_state['class_mean']:
        df['Height'] = df.groupby('Sex')['Height'].transform(lambda x: x.fillna(round(x.mean(), 2)))
        df['Weight'] = df.groupby('Sex')['Weight'].transform(lambda x: x.fillna(round(x.mean(), 2)))

    if st.session_state['class_median']:
        df['Height'] = df.groupby('Sex')['Height'].transform(lambda x: x.fillna(round(x.median(), 2)))
        df['Weight'] = df.groupby('Sex')['Weight'].transform(lambda x: x.fillna(round(x.median(), 2)))

    if st.session_state['class_mode']:
        df['Height'] = df.groupby('Sex')['Height'].transform(lambda x: x.fillna(round(x.mode().mean(), 2)))
        df['Weight'] = df.groupby('Sex')['Weight'].transform(lambda x: x.fillna(round(x.mode().mean(), 2)))

with col2:
    st.dataframe(
        df[:10].style.apply(color_nones_male, axis=None),
        hide_index = True, 
        use_container_width = True,
    )

with col3:
    st.dataframe(
        df[10:].style.apply(color_nones_female, axis=None), 
        hide_index = True, 
        use_container_width = True,
    )

df = df.reset_index(drop=True)
h, w = 170, 80

distances = [round(np.sqrt((h-df['Height'][i])**2+(w-df['Weight'][i])**2), 2) for i in df.index]
k_indices = np.argsort(distances)[:5]

k_labels = [df['Sex'][i] for i in k_indices]
male_count, female_count = k_labels.count('Male'), k_labels.count('Female')
prediction = 'Male' if male_count>female_count else 'Female' if male_count<female_count else 'Unknown'

col1, col2 = st.columns([3, 1])

with col2:  
    col21, col22 = st.columns(2)
    with col21:
        st.metric('Male', f"{male_count}/5")
    with col22:
        st.metric('Female', f"{female_count}/5")
    st.write(f"Predicted Sex: {prediction}")

with col1:
    fig = go.Figure()
    for i in k_indices[:5]:
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
        customdata = distances[:10],
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
        x = [h],
        y = [w],
        mode = 'markers',
        marker_color = 'black',
        marker_symbol = 'x',
        marker_size = 10,
        name = 'To Predict',
    ))
    fig.update_layout(
        margin = dict(t=0, r=20, b=0),
        height = 300,
        showlegend = False,
        xaxis_title = 'Height (cm)',
        yaxis_title = 'Weight (kg)',
    )
    st.plotly_chart(fig, config=config)

# ----------------------------------
st.divider()
st.subheader('Unbalanced Data')
# ----------------------------------

df = pd.DataFrame({
    'Enabled': [True]*12 + [False]*8,
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

column_config = {
    'Enabled': st.column_config.CheckboxColumn(label='Enabled'),
    'Sex': st.column_config.Column(disabled=True),
    'Height': st.column_config.Column(disabled=True),
    'Weight': st.column_config.Column(disabled=True),
}

col1, col2 = st.columns(2)

with col1:
    df[df['Sex']=='Male'] = st.data_editor(
        df[df['Sex']=='Male'], 
        column_config = column_config, 
        hide_index = True, 
        use_container_width = True
    )

with col2:
    df[df['Sex']=='Female'] = st.data_editor(
        df[df['Sex']=='Female'], 
        column_config = column_config, 
        hide_index = True, 
        use_container_width = True
    )

filtered_df = df[df['Enabled']==True]

filtered_df = filtered_df.reset_index(drop=True)
h, w = 170, 80
distances = [round(np.sqrt((h-filtered_df['Height'][i])**2+(w-filtered_df['Weight'][i])**2), 2) for i in filtered_df.index]
k_indices = np.argsort(distances)[:5]

k_labels = [filtered_df['Sex'][i] for i in k_indices]
male_count, female_count = k_labels.count('Male'), k_labels.count('Female')
prediction = 'Male' if male_count>female_count else 'Female' if male_count<female_count else 'Unknown'

col1, col2 = st.columns([3, 1])

with col2:  
    col21, col22 = st.columns(2)
    with col21:
        st.metric('Male', f"{male_count}/5")
    with col22:
        st.metric('Female', f"{female_count}/5")
    st.write(f"Predicted Sex: {prediction}")

with col1:
    fig = go.Figure()
    for i in k_indices[:5]:
        fig.add_trace(go.Scatter(
            x = [filtered_df['Height'][i]],
            y = [filtered_df['Weight'][i]],
            mode = 'markers',
            marker_color = 'yellow',
            marker_size = 20,
            name = 'k-nearest',
            hoverinfo = 'skip',
        ))
    fig.add_trace(go.Scatter(
        x = filtered_df[filtered_df['Sex']=='Male']['Height'],
        y = filtered_df[filtered_df['Sex']=='Male']['Weight'],
        customdata = distances[:10],
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
        x = filtered_df[filtered_df['Sex']=='Female']['Height'],
        y = filtered_df[filtered_df['Sex']=='Female']['Weight'],
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
        x = [h],
        y = [w],
        mode = 'markers',
        marker_color = 'black',
        marker_symbol = 'x',
        marker_size = 10,
        name = 'To Predict',
    ))
    fig.update_layout(
        margin = dict(t=0, r=20, b=0),
        height = 300,
        showlegend = False,
        xaxis_title = 'Height (cm)',
        yaxis_title = 'Weight (kg)',
    )
    st.plotly_chart(fig, config=config)