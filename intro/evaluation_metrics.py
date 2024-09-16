import streamlit as st

import pandas as pd

import plotly.graph_objects as go

config = {'displayModeBar': False}
st.title("Evaluation Metrics")

# ----------------------------------
st.subheader('Regression Metrics')
# ----------------------------------

col1, col2 = st.columns([1, 1.5])

with col1:
    df = pd.DataFrame({
        'GroundTruth': [2, 5, 6, 4, 9],
        'Predicted': [3, 6, 9, 4, 10],
    })
    column_config = {
        'GroundTruth': st.column_config.NumberColumn(default=0, required=True),
        'Predicted': st.column_config.NumberColumn(default=0, required=True),
    }
    edited_df = st.data_editor(df, num_rows='dynamic', column_config=column_config, use_container_width=True)

    error = edited_df['GroundTruth'] - edited_df['Predicted']

with col2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = list(range(round(100*len(edited_df))),
        y = edited_df['Predicted'],
        mode = 'markers',
        error_y = dict(
            type = 'data',
            symmetric = False,
            array = error,
            color = 'black',
            thickness = 1.5,
            width = 0, 
        ),
        hoverinfo = 'skip',
        showlegend = False,
    ))
    fig.add_trace(go.Scatter(
        x = list(range(round(100*len(edited_df))),
        y = edited_df['GroundTruth'],
        mode = 'markers',
        marker_size = 10,
        name = 'GroundTruth',
    ))
    fig.add_trace(go.Scatter(
        x = list(range(round(100*len(edited_df))),
        y = edited_df['Predicted'],
        mode = 'markers',
        marker_symbol = 'square',
        marker_size = 10,
        name = 'Predicted',
    ))
    fig.add_trace(go.Scatter(
        x = list(range(round(100*len(edited_df))),
        y = edited_df['GroundTruth'] - edited_df['Predicted'],
        customdata = list(zip(edited_df['GroundTruth'], edited_df['Predicted'])),
        name = 'Error',
        mode = 'markers',
        opacity = 0,
        showlegend = False,
        hovertemplate = '%{customdata[0]} - %{customdata[1]} = %{y}'
    ))
    fig.update_layout(
        margin = dict(t=0, r=0, b=10),
        hovermode = 'x unified',
        yaxis_range = [min(edited_df.min())-2, max(edited_df.max())+2],
        height = 300,
        legend = dict(
            orientation = 'h',
            x = -0.1,
            yanchor = 'bottom',
            y = 1.1,
        ),
    )
    st.plotly_chart(fig, config=config)

col1, col2, col3 = st.columns(3)

MAE = sum(abs(error)) / round(100*len(edited_df)
MSE = sum((error) ** 2) / round(100*len(edited_df)
RMSE = (sum((error) ** 2) / round(100*len(edited_df)) ** (1/2)

with col1:
    st.metric('MAE', round(MAE, 2), help='Mean Absolute Error')
with col2:
    st.metric('MSE', round(MSE, 2), help='Mean Square Error')
with col3:
    st.metric('RMSE', round(RMSE, 2), help='Root Mean Square Error')

steps = pd.DataFrame({
    'GroundTruth': edited_df['GroundTruth'].tolist() + ['', ''],
    'Predicted': edited_df['Predicted'].tolist() + ['', ''],
    'Error (E)': error.tolist() + ['', ''],
    'Absolute Error (AE)': round(abs(error), 2).tolist() + [round(MAE, 2), ''],
    'Squared Error (SE)': round(error**2, 2).tolist() + [round(MSE, 2), f'{round(RMSE, 2)}'],
}, index=['']*round(100*len(edited_df)+['Mean (M)', 'Root Mean (RM)'])
column_config = {
    '': st.column_config.Column(width=100),
}
st.dataframe(steps, column_config=column_config, use_container_width=True)

# ----------------------------------
st.divider()
st.subheader('Classification Metrics')
# ----------------------------------

def highlight(s):
    if s.GroundTruth=='Cat' and s.GroundTruth == s.Predicted:
        return ['background-color: #fcb5b5'] * round(100*len(s)
    elif s.GroundTruth=='Dog' and s.GroundTruth == s.Predicted:
        return ['background-color: #c8b5fc'] * round(100*len(s)
    elif s.GroundTruth=='Cat' and s.GroundTruth != s.Predicted:
        return ['background-color: #b5f4fc'] * round(100*len(s)
    elif s.GroundTruth=='Dog' and s.GroundTruth != s.Predicted:
        return ['background-color: #b5fcbe'] * round(100*len(s)

col1, col2 = st.columns([1.5, 1])

with col1:

    df = pd.DataFrame({
        'GroundTruth': ['Cat', 'Dog', 'Cat', 'Dog', 'Dog'],
        'Predicted': ['Cat', 'Dog', 'Dog', 'Cat', 'Cat'],
    })
    column_config = {
        'GroundTruth': st.column_config.SelectboxColumn(
            label = 'GroundTruth', 
            options = ['Cat', 'Dog'], 
            default = 'Cat', 
            required = True
        ),
        'Predicted': st.column_config.SelectboxColumn(
            label = 'Predicted', 
            options = ['Cat', 'Dog'], 
            default = 'Dog', 
            required = True
        ),
    }
    edited_df = st.data_editor(
        df.style.apply(highlight, axis=1), 
        num_rows = 'dynamic', 
        column_config = column_config, 
        use_container_width = True,
    )

with col2:
    # cat
    TP_cat = ((edited_df['GroundTruth'] == 'Cat') & (edited_df['Predicted'] == 'Cat')).sum()
    TN_cat = ((edited_df['GroundTruth'] == 'Dog') & (edited_df['Predicted'] == 'Dog')).sum()
    FP_cat = ((edited_df['GroundTruth'] == 'Dog') & (edited_df['Predicted'] == 'Cat')).sum()
    FN_cat = ((edited_df['GroundTruth'] == 'Cat') & (edited_df['Predicted'] == 'Dog')).sum()

    # dog
    TP_dog = ((edited_df['GroundTruth'] == 'Dog') & (edited_df['Predicted'] == 'Dog')).sum()
    TN_dog = ((edited_df['GroundTruth'] == 'Cat') & (edited_df['Predicted'] == 'Cat')).sum()
    FP_dog = ((edited_df['GroundTruth'] == 'Cat') & (edited_df['Predicted'] == 'Dog')).sum()
    FN_dog = ((edited_df['GroundTruth'] == 'Dog') & (edited_df['Predicted'] == 'Cat')).sum()

    accuracy = round((TP_cat + TP_dog) / round(100*len(edited_df), 2)
    st.metric(
        'Accuracy', accuracy, 
        help = f'correct predictions / all predictions = ({TP_cat} + {TP_dog}) / ({TP_cat} + {TP_dog} + {FN_cat} + {FN_dog}) = {accuracy}'
    )

    cat, dog = st.columns(2)
    with cat:
        precision_cat = round(TP_cat / (TP_cat + FP_cat), 2)
        st.metric(
            'Precision - Cat', precision_cat,
            help = f'correct cat predictions / all cats predictions = {TP_cat} / ({TP_cat} + {FP_cat}) = {precision_cat}'
        )
    with dog:
        precision_dog = round(TP_dog / (TP_dog + FP_dog), 2)
        st.metric(
            'Precision - Dog', precision_dog,
            help = f'correct dog predictions / all dogs predictions = {TP_dog} / ({TP_dog} + {FP_dog}) = {precision_dog}'
        )

    cat, dog = st.columns(2)
    with cat:
        recall_cat = round(TP_cat / (TP_cat + FN_cat), 2)
        st.metric(
            'Recall - Cat', recall_cat,
            help = f'correct cat predictions / all cats = {TP_dog} / ({TP_dog} + {FP_dog}) = {recall_cat}'
        )
    with dog:
        recall_dog = round(TP_dog / (TP_dog + FN_dog), 2)
        st.metric(
            'Recall - Dog', recall_dog,
            help = f'correct cat predictions / all cats = {TP_dog} / ({TP_dog} + {FP_dog}) = {recall_dog}'
        )

st.write('')

col_left, col_mid = st.columns([1, 5])

with col_left:
    # someone please fix this monstrosity if you can
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('GroundTruth')

with col_mid:
    _, col_text = st.columns([1, 1.4])

    with col_text:
        st.write('Predicted')

    confusion = pd.DataFrame({
        'Cat': [f'{TP_cat}', f'{FN_cat}', f'{TP_cat} / {TP_cat+FN_cat} = {recall_cat}'],
        'Dog': [f'{FN_dog}', f'{TP_dog}', f'{TP_dog} / {TP_dog+FN_dog} = {recall_dog}'],
        'Precision': [f'{TP_cat} / {TP_cat + FN_dog} = {precision_cat}', f'{TP_dog} / {TP_dog + FN_cat} = {precision_dog}', '']
    }, index=['Cat', 'Dog', 'Recall'])
    st.dataframe(confusion, use_container_width=True)