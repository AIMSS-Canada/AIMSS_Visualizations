import streamlit as st
from .demos.data_preprocessing import dp
from .demos.evaluation_metrics import eval

st.set_page_config(layout="centered")

preprocessing_tab, eval_metrics_tab = st.tabs(["Data Preprocessing", "Evaluation Metrics"])

with preprocessing_tab:
    dp()
with eval_metrics_tab:
    eval()
