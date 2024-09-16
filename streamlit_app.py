import streamlit as st

# ----------------------------------
# Home

welcome = st.Page(
    'home/welcome.py', 
    title = 'Welcome',
    icon = ':material/home:', 
    default = True,
)

# ----------------------------------
# Intro to ML

eval_metrics_page = st.Page(
    'intro/evaluation_metrics.py', 
    title = 'Evaluation Metrics',
    icon = ':material/analytics:',
)
preprocessing = st.Page(
    'intro/data_preprocessing.py', 
    title = 'Data Preprocessing',
    icon = ':material/query_stats:',
)

# ----------------------------------
# Model Algorithms

kmeans = st.Page(
    'model_algos/kmeans.py', 
    title = 'k-Means Clustering',
    icon = ':material/linked_services:',
)
dt = st.Page(
    'model_algos/decision_trees.py', 
    title = 'Decision Trees',
    icon = ':material/account_tree:',
)
linr = st.Page(
    'model_algos/linear_regression.py', 
    title = 'Linear Regression',
    icon = ':material/pen_size_3:',
)
logr = st.Page(
    'model_algos/logistic_regression.py', 
    title = 'Logistic Regression',
    icon = ':material/heat:',
)
knn = st.Page(
    'model_algos/knn.py', 
    title = 'k-Nearest Neighbours',
    icon = ':material/circles_ext:',
)
fit = st.Page(
    'model_algos/model_fitting.py', 
    title = 'Model Fitting',
    icon = ':material/troubleshoot:',
)

# ----------------------------------
# Chest Xray NN

xray_data = st.Page(
    'xray_nn/data.py', 
    title = 'Data',
    icon = ':material/rib_cage:',
)
xray_model = st.Page(
    'xray_nn/model.py', 
    title = 'Model',
    icon = ':material/share:',
)
xray_predict = st.Page(
    'xray_nn/predict.py', 
    title = 'Predict',
    icon = ':material/subject:',
)

# ----------------------------------

pg = st.navigation({
    'Home': [welcome],
    'Intro to ML': [eval_metrics_page, preprocessing],
    'Model Algorithms': [kmeans, dt, linr, logr, knn, fit],
    'Chest X-ray Classification': [xray_data, xray_model, xray_predict],
})
pg.run()