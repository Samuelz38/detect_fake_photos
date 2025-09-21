import streamlit as st

def render_home_page():
    st.title("🔍 Bem-vindo ao V.A.I (Verificador de Autenticidade da Informação)")
    st.write("Compare e visualize possíveis modificações em dados de forma simples e rápida.")

    st.markdown("---") 

    col1, col2 = st.columns(2)


    with col1:
        with st.container(border=True):
            st.subheader("📊 Verificar Imagem")
            st.write("Envie uma imagem para análise e verificação de alterações.")
            if st.button("Prosseguir", key="btn_prosseguir"):
                st.switch_page("pages/result.py")

 
    with col2:
        with st.container(border=True):
            st.subheader("🎥 Verificar Vídeo")
            st.write("Envie um video para análise e verificação de alterações.")
            if st.button("Prosseguir", key="btn_mural"):
                st.switch_page("pages/comp_video.py")



