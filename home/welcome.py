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

st.title("Hello")
st.write("Welcome to the 2024-2025 Artificial Intelligence in Medical Systems Society's Intro to AI in Medical Applications Course!")
st.write("This is a little hub of how some simple models work. Since it's a new addition to the course, if there are any suggestions or issues, please let us know!")

st.header("The Team")
st.write("**Instructors**: Bahareh Behroozi Asl, Aminreza Khandan, Golnaz Mesbahi, Mohammad Reza Taesiri")
st.write("**Leads**: Micheal Xie, Ying Wan")
st.write("**Special Thanks**: Shane Eaton, Ehsan Misaghi")


