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
# 1. Intro to AI and Data

recording_1 = st.Page(
    'intro/1-recording.py', 
    title = 'Recording',
    icon = ':material/video_library:',
    url_path = '1-recording',
)
activities_1 = st.Page(
    'intro/1-activities.py', 
    title = 'Activities',
    icon = ':material/analytics:',
    url_path = '1-activities',
)
resources_1 = st.Page(
    'intro/1-resources.py', 
    title = 'Resources',
    icon = ':material/bookmark_border:',
    url_path = '1-resources',
)

# ----------------------------------
# 2. ML Algorithms

recording_2 = st.Page(
    'ml_algos/2-recording.py', 
    title = 'Recording',
    icon = ':material/video_library:',
    url_path = '2-recording',
)
activities_2 = st.Page(
    'ml_algos/2-activities.py', 
    title = 'Activities',
    icon = ':material/analytics:',
    url_path = '2-activities',
)
resources_2 = st.Page(
    'ml_algos/2-resources.py', 
    title = 'Resources',
    icon = ':material/bookmark_border:',
    url_path = '2-resources',
)

# ----------------------------------
# 3. DL Algorithms

recording_3 = st.Page(
    'ai_algos/3-recording.py', 
    title = 'Recording',
    icon = ':material/video_library:',
    url_path = '3-recording',
)
resources_3 = st.Page(
    'ai_algos/3-resources.py', 
    title = 'Resources',
    icon = ':material/bookmark_border:',
    url_path = '3-resources',
)

# ----------------------------------
# 4. LLMs

recording_4 = st.Page(
    'llms/4-recording.py', 
    title = 'Recording',
    icon = ':material/video_library:',
    url_path = '4-recording',
)
resources_4 = st.Page(
    'llms/4-resources.py', 
    title = 'Resources',
    icon = ':material/bookmark_border:',
    url_path = '4-resources',
)

# ----------------------------------
# 5. How to Read an AI Paper

recording_5 = st.Page(
    'ai_paper/5-recording.py', 
    title = 'Recording',
    icon = ':material/video_library:',
    url_path = '5-recording',
)
resources_5 = st.Page(
    'ai_paper/5-resources.py', 
    title = 'Resources',
    icon = ':material/bookmark_border:',
    url_path = '5-resources',
)

# ----------------------------------
# 6. AI Applications

# ----------------------------------
# Extra Resources

# ----------------------------------
# Assignment
assignment = st.Page(
    'Assignment/assignment.py', 
    title = 'Assignment',
    icon = ':material/assignment:',
)

# ----------------------------------
# Chest Xray NN

# xray_data = st.Page(
#     'xray_nn/data.py', 
#     title = 'Data',
#     icon = ':material/rib_cage:',
# )
# loss = st.Page(
#     'xray_nn/model_loss.py', 
#     title = 'Model Loss',
#     icon = ':material/share:',
# )
# tuning = st.Page(
#     'xray_nn/model_tuning.py', 
#     title = 'Model Tuning',
#     icon = ':material/build:',
# )
# xray_predict = st.Page(
#     'xray_nn/predict.py', 
#     title = 'Predict',
#     icon = ':material/subject:',
# )

# ----------------------------------

if  __name__ == "__main__":
    pg = st.navigation({
        'Home': [welcome],
        '1. Intro to AI and Data': [recording_1, activities_1, resources_1],
        '2. Machine Learning Algorithms': [recording_2, activities_2, resources_2],
        '3. Deep Learning Algorithms': [recording_3, resources_3],
        '4. Large Language Models': [recording_4, resources_4],
        '5. How to Read an AI Paper': [recording_5, resources_5],
        '6. AI Applications': [],
        'Extra Resources': [],
        'Assignment': [assignment]
        # 'Chest X-ray Classification': [xray_data, loss, tuning, xray_predict],
    }, expanded=True)
    pg.run()
    st.logo("./src/49018_AIMSS_RB_GR_MV-05.webp", size="large", link="https://www.aimss.ca")
