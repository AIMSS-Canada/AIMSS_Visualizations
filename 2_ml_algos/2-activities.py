import streamlit as st
from ml_algos.demos.linear_regression import linr
from ml_algos.demos.logistic_regression import logr
from ml_algos.demos.knn import knn
from ml_algos.demos.kmeans import kmeans
from ml_algos.demos.decision_trees import dt

st.set_page_config(layout="wide")

linr_tab, logr_tab, knn_tab, kmeans_tab, dt_tab = st.tabs(["Linear Regression", "Logistic Regression", "K-Nearest Neighbors", "k-Means", "Decision Trees"])

with linr_tab:
    linr()
with logr_tab:
    logr()
with knn_tab:
    knn()
with kmeans_tab:
    kmeans()
with dt_tab:
    dt()
