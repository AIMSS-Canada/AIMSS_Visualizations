import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="centered")
config = {'displayModeBar': False}
st.title("k-Means")

with st.expander('Help'):
    st.write('''
    The points illustrate the height and weight of 20 boys and 20 girls that were artificially generated. The ground truth labels are not given in this plot.
             
    K-means produces clusters that usually form a circular or globby shape. This is because it optimizes the distance between each point in a cluster and
    its centroid. This is useful in some cases in which similar points lie near eachother, but in others, it may be helpful to look into other clustering 
    algorithms.
             
    **Hint:** k affects the number of clusters (in different colors) to create. In this case, since the data is split into male and female, typically we
    would cluster into 2 (k=2), but you are free to create as many clusters as you'd like.
    ''')


st.error('Note: This is a bad use case for k-means (non circular clustered points), maybe use a different datset as example')
def euclidean_dist(x, y):
    return np.sqrt((st.session_state['height'] - x) ** 2 + (st.session_state['weight'] - y) ** 2)

df = pd.DataFrame({
    'Sex': ['Male']*20 + ['Female']*20,
    'Height': [
        169.2, 173.29, 174.1, 164.54, 162.70, 174.9, 175.1, 172.56, 176.02, 179.3,   # Male
        180.5, 165.3, 177.1, 168.9, 170.2, 175.0, 167.5, 171.8, 179.0, 181.2,        # Male
        155.8, 168.7, 174.2, 167.5, 169.1, 162.3, 155.0, 167.8, 170.5, 166.0,        # Female
        163.89, 152.42, 170.32, 160.6, 164.98, 172.2, 161.41, 171.53, 172.2, 163.4,  # Female
    ],
    'Weight': [
        79.6, 79.94, 84.2, 82.8, 83.01, 78.65, 84.21, 84.7, 82.98, 85.21,   # Male
        90.1, 85.5, 88.0, 82.3, 87.2, 80.4, 83.7, 88.5, 89.2, 92.3,         # Male
        65.4, 70.2, 72.5, 69.1, 73.2, 68.4, 64.3, 71.6, 74.1, 67.8,         # Female
        76.23, 77.87, 79.6, 78.86, 79.1, 79.56, 76.69, 80.4, 81.63, 82.15,  # Female
    ],
})

# scale_btn = st.checkbox('Scale Data', key='scale')
st.session_state['scale'] = False

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
        min_value = 1, 
        max_value = 40, 
        value = 2, 
        key = 'k'
    )

    features = df[['Height', 'Weight']]

    kmeans = KMeans(n_clusters=st.session_state['k'], init='k-means++', max_iter=300, n_init=10, random_state=0)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    df['Cluster'] = kmeans.fit_predict(scaled_features)

    pt = [[st.session_state['height'], st.session_state['weight']]]

    scaled_pt = scaler.transform(pt)
    cluster_prediction = kmeans.predict(scaled_pt)

    st.write(f'Predicted Cluster: {cluster_prediction[0]+1}')

    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    centers_df = pd.DataFrame(centers, columns=['Height', 'Weight'])
    centers_df['Cluster'] = range(st.session_state['k'])

    colors = [
        "#ff006e", "#fb5607", "#8338ec", "#3a86ff", "#ffbe0b", "#3a0ca3", "#e27396", "#31572c",
        "#5e548e", "#4f772d", "#edc531", "#FF007F", "#FF4C4C", "#FF9A4C", "#FFEC4C", "#9AFF4C",
        "#4CFF4C", "#4CFF9A", "#4CFFFF", "#4C7FFF", "#4C00FF", "#7F4CFF", "#FF4CFF", "#FF4C7F",
        "#FF7F7F", "#FFB27F", "#FFEC7F", "#B2FF7F", "#7FFF7F", "#7FFFB2", "#7FFFFF", "#7F7FFF",
        "#7F00FF", "#BF7FFF", "#FF7FBF", "#FF7F7F", "#FFB2B2", "#FFC4B2", "#FFDAB2", "#B2FFB2",
        "#B2FFC4", "#B2FFFF"
    ]
    if st.session_state['k'] > 1:
        cluster_colors = colors[:st.session_state['k']]
    else:
        cluster_colors = [colors[0], colors[0]]

with col1:    
    fig = go.Figure()
    for i, centroid in centers_df.iterrows():
        fig.add_trace(go.Scatter(
            x=[centroid['Height']],
            y=[centroid['Weight']],
            mode='markers',
            marker = dict(
                color = cluster_colors[i], 
                size = 10, 
                symbol = 'x'
            ),
            name=f'Centroid {i+1}'
        ))
    fig.add_trace(go.Scatter(
        x = df['Height'],
        y = df['Weight'],
        customdata = df['Cluster']+1,
        marker_color = df['Cluster'],
        marker_colorscale = cluster_colors,
        mode = 'markers',
        hovertemplate = '''
            Cluster: %{customdata}<br>
            Height: %{x}<br>
            Weight: %{y}<extra></extra>
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
