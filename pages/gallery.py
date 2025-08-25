import streamlit as st
from src.layout.gallery_comparison import render_gallery_page

st.set_page_config(
    page_title="Mural de Imagens",
    page_icon="🖼️",
    layout="wide"
)

render_gallery_page()
