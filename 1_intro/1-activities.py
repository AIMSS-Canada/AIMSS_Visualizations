import streamlit as st
from 1_intro.demos.data_preprocessing import dp
from 1_intro.demos. evaluation_metrics import eval

st.set_page_config(layout="centered")

preprocessing_tab, eval_metrics_tab = st.tabs(["Data Preprocessing", "Evaluation Metrics"])

with preprocessing_tab:
    dp()
with eval_metrics_tab:
    eval()
