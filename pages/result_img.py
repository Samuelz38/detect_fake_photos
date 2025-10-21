import streamlit as st
from src.layout.screen_layout import render_result_page

st.set_page_config(
    page_title="Comparação de Imagens",
    page_icon="🖼️",
    layout="wide"
)

render_result_page()
