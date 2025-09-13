import streamlit as st
from src.layout.video_comp import render_result_page_video

st.set_page_config(
    page_title="Comparação de Videos",
    page_icon="🎥",
    layout="wide"
)

render_result_page_video()