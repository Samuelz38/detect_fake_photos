import streamlit as st
from src.layout.home import render_home_page

def main():
    st.set_page_config(
        page_title="Meu App de Imagens",
        page_icon="🖼️",
        layout="centered"
    )
    
    render_home_page()

if __name__ == "__main__":
    main()