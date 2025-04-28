import streamlit as st
import requests
from streamlit_lottie import st_lottie

@st.cache_data(show_spinner=False)
def load_lottie(url: str):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None

st.set_page_config(page_title="4th-Down Success Explorer", layout="centered")
st.markdown(
    """
    <style>
    .stApp { background-color: #000000; color: #FFFFFF; }
    h1 { font-size: 3rem; text-align: center; margin-bottom: 0.5rem; }
    .stMarkdown p { font-size: 1.1rem; color: #EEEEEE; }
    .stButton>button {
        background-color: #1F77B4;
        color: #FFFFFF;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #155A8A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

anim = load_lottie("https://assets7.lottiefiles.com/packages/lf20_x62chJ.json")
if anim:
    st_lottie(anim, height=120, width=200)

st.title("🏈 4th-Down Success Explorer")
st.markdown(
    """
    Welcome! Explore our two 4th-down success models and compare them:

    - **Without Team Coefficients**  
    - **With Team Coefficients**  
    - **Compare 4th-Down**  

    Click one of the buttons below to go to the desired page.
    """
)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    if st.button("🚀 Without Team Coefficients"):
        if hasattr(st, "switch_page"):
            try:
                st.switch_page("pages/1_4th_Down_Without_Coefficients.py")
            except Exception:
                pass
        st.query_params = {"page": "1_4th_down_without_coefficients"}

with col2:
    if st.button("🚀 With Team Coefficients"):
        if hasattr(st, "switch_page"):
            try:
                st.switch_page("pages/2_4th_Down_With_Coefficients.py")
            except Exception:
                pass
        st.query_params = {"page": "2_4th_down_with_coefficients"}

with col3:
    if st.button("🚀 Compare 4th-Down"):
        if hasattr(st, "switch_page"):
            try:
                st.switch_page("pages/3_compare_4th_down.py")
            except Exception:
                pass
        st.query_params = {"page": "3_compare_4th_down"}

st.markdown("---")
st.markdown("© 2025 4th-Down Success Explorer")