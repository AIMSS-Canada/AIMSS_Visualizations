import streamlit as st

st.set_page_config(layout="centered")

_, mid, _ = st.columns([3, 1, 3])
with mid:
    try:
        image_path = "https://static.wixstatic.com/media/ebe019_946bf08cf70d4d24aed5d90bcdf8b0f8~mv2.png/v1/fill/w_270,h_270,al_c,q_85,usm_0.66_1.00_0.01,enc_auto/AIMSS-whiteback-darkteal.png"
        url = "https://www.aimss.ca"
        st.markdown(f'<a href="{url}" target="_blank"><img src="{image_path}" alt="Image" style="width:100%;"></a>', unsafe_allow_html=True)
    except:
        st.image("./src/AIMSS-whiteback-darkteal.webp")

st.title("AI in Healthcare Course 25/26")

session_data = {
    "Session": [
        "Intro to AI and Data",
        "ML Algorithms",
        "Deep Learning Algorithms",
        "Large Language Models",
        "How to Read an AI Paper",
        "Applications of AI and Research",
    ],
    "Date": [
        "Nov. 3, 2025", 
        "Dec. 1, 2025", 
        "Jan. 5, 2026",
        "Feb. 2, 2026",
        "Mar. 2, 2026",
        "Mar. 30, 2026",
    ],
    "Instructor": [
        "Nazila Ameli", 
        "Mahdieh Mallahnezhad", 
        "Golnaz Mesbahi",
        "Sacha Davis",
        "Golnaz Mesbahi",
        "Ehsan Misaghi",
    ],
}
st.dataframe(session_data)

st.write("**Acknowledgements**")
st.write("Instructors: Mahdieh Mallahnezhad, Nazila Ameli, Golnaz Mesbahi, Sacha Davis")
st.write("Support: Ehsan Misaghi, Micheal Xie, Ying Wan, Shane Eaton")