import streamlit as st
import importlib
dp = importlib.import_module("1_intro.demos.data_preprocessing")
eval = importlib.import_module("1_intro.demos.evaluation_metrics")

st.set_page_config(layout="centered")

preprocessing_tab, eval_metrics_tab = st.tabs(["Data Preprocessing", "Evaluation Metrics"])

with preprocessing_tab:
    dp()
with eval_metrics_tab:
    eval()
