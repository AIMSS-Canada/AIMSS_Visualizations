import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
st.set_page_config(layout="wide")

config = {'displayModeBar': False}
st.title("Decision Trees")

st.write('''
    The goal for a decision tree is to split the data into buckets that are mostly homogenous in class. Decision trees are
    normally deeper (having more branching conditions) than this example, but it gives you an idea of how the branches work 
    and how predictions are made.
''')

def cont_filter(df, key, comparison, value):
    if comparison == '<':
        return df[df[key]<value], df[df[key]>=value]
    if comparison == '<=':
        return df[df[key]<=value], df[df[key]>value]
    if comparison == '>':
        return df[df[key]>value], df[df[key]<=value]
    if comparison == '>=':
        return df[df[key]>=value], df[df[key]<value]
    
def bin_filter(df, key, comparison, value):
    if comparison == 'is':
        return df[df[key].isin(value)], df[~df[key].isin(value)]
    if comparison == 'not':
        return df[~df[key].isin(value)], df[df[key].isin(value)]

df = pd.read_csv('./data/heart.csv')

cols = df.columns
cat_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
target = df['HeartDisease']

st.dataframe(df[:5], hide_index=True)

_, col2, _ = st.columns([0.9, 3, 1])

with col2:
    col11, col12, col13 = st.columns([1.5,0.5,1.5])
    with col11:
        st.selectbox('', options=cols, key='C', label_visibility='hidden')
    with col12:
        if st.session_state['C'] not in cat_cols:
            comp = st.selectbox('', options=['<', '<=', '>=', '>'], key='C_cat_comp', label_visibility='hidden')
            with col13:
                val = st.number_input('', step=1, key='C_cont_val', label_visibility='hidden')

            df_C_true, df_C_false = cont_filter(df, st.session_state['C'], comp, val)

        else:
            comp = st.selectbox('', options=['is', 'not'], key='C_cont_comp', label_visibility='hidden')
            with col13:
                vals = st.multiselect('', options=set(df[st.session_state['C']]), default=df[st.session_state['C']][0], key='C_cat_val', label_visibility='hidden')

            df_C_true, df_C_false = bin_filter(df, st.session_state['C'], comp, vals)

col_true, _, col_false = st.columns([2,0.2,2])
with col_true:
    st.subheader('True')
    col1, col2 = st.columns(2)
    with col1:
        st.metric('no CVD: ', f"{len(df_C_true[df_C_true['HeartDisease']==0])}/{len(df_C_true)}")
    with col2:
        st.metric('CVD: ', f"{len(df_C_true[df_C_true['HeartDisease']==1])}/{len(df_C_true)}")

with col_false:
    st.subheader('False')
    col1, col2 = st.columns(2)
    with col1:
        st.metric('no CVD: ', f"{len(df_C_false[df_C_false['HeartDisease']==0])}/{len(df_C_false)}")
    with col2:
        st.metric('CVD: ', f"{len(df_C_false[df_C_false['HeartDisease']==1])}/{len(df_C_false)}")

col1, _, col2 = st.columns([2,0.2,2])

with col1:
    col11, col12, col13 = st.columns([1.5,0.7,1.5])
    with col11:
        st.selectbox('', options=cols, key='L', label_visibility='hidden')
    with col12:
        if st.session_state['L'] not in cat_cols:
            comp = st.selectbox('', options=['<', '<=', '>=', '>'], key='L_cat_comp', label_visibility='hidden')
            with col13:
                val = st.number_input('', step=1, key='L_cont_val', label_visibility='hidden')

            df_L_true, df_L_false = cont_filter(df_C_true, st.session_state['L'], comp, val)

        else:
            comp = st.selectbox('', options=['is', 'not'], key='L_cont_comp', label_visibility='hidden')
            with col13:
                vals = st.multiselect('', options=set(df[st.session_state['L']]), default=df[st.session_state['L']][0], key='L_cat_val', label_visibility='hidden')
    
            df_L_true, df_L_false = bin_filter(df_C_true, st.session_state['L'], comp, vals)

with col2:
    col11, col12, col13 = st.columns([1.5,0.7,1.5])
    with col11:
        st.selectbox('', options=cols, key='R', label_visibility='hidden')
    with col12:
        if st.session_state['R'] not in cat_cols:
            comp = st.selectbox('', options=['<', '<=', '>=', '>'], key='R_cont_comp', label_visibility='hidden')
            with col13:
                val = st.number_input('', step=1, key='R_cont_val', label_visibility='hidden')

            df_R_true, df_R_false = cont_filter(df_C_false, st.session_state['R'], comp, val)

        else:
            comp = st.selectbox('', options=['is', 'not'], key='R_cat_comp', label_visibility='hidden')
            with col13:
                vals = st.multiselect('', options=set(df[st.session_state['R']]), default=df[st.session_state['R']][0], key='R_cat_val', label_visibility='hidden')
    
            df_R_true, df_R_false = bin_filter(df_C_false, st.session_state['R'], comp, vals)

col_L_true, col_L_false, _, col_R_true, col_R_false = st.columns([1,1,0.2,1,1])
with col_L_true:
    st.subheader('True')
    st.metric('no CVD: ', f"{len(df_L_true[df_L_true['HeartDisease']==0])}/{len(df_L_true)}")
    st.metric('CVD: ', f"{len(df_L_true[df_L_true['HeartDisease']==1])}/{len(df_L_true)}")

with col_L_false:
    st.subheader('False')
    st.metric('no CVD: ', f"{len(df_L_false[df_L_false['HeartDisease']==0])}/{len(df_L_false)}")
    st.metric('CVD: ', f"{len(df_L_false[df_L_false['HeartDisease']==1])}/{len(df_L_false)}")

with col_R_true:
    st.subheader('True')
    st.metric('no CVD: ', f"{len(df_R_true[df_R_true['HeartDisease']==0])}/{len(df_R_true)}")
    st.metric('CVD: ', f"{len(df_R_true[df_R_true['HeartDisease']==1])}/{len(df_R_true)}")

with col_R_false:
    st.subheader('False')
    st.metric('no CVD: ', f"{len(df_R_false[df_R_false['HeartDisease']==0])}/{len(df_R_false)}")
    st.metric('CVD: ', f"{len(df_R_false[df_R_false['HeartDisease']==1])}/{len(df_R_false)}")