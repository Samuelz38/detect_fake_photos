import streamlit as st

def render_home_page():
    st.title("🔍 Bem-vindo ao Verificador de Imagens")
    st.write("Compare e visualize possíveis modificações em imagens de forma simples e rápida.")

    st.markdown("---")  # Linha divisória

    # Criar duas colunas para os dois containers
    col1, col2 = st.columns(2)

    # ===== Container 1: Prosseguir =====
    with col1:
        with st.container(border=True):
            st.subheader("📊 Verificar Imagem")
            st.write("Envie uma imagem para análise e verificação de alterações.")
            if st.button("Prosseguir", key="btn_prosseguir"):
                st.switch_page("pages/result.py")

    # ===== Container 2: Mural =====
    with col2:
        with st.container(border=True):
            st.subheader("🎥 Verificar Vídeo")
            st.write("Envie um video para análise e verificação de alterações.")
            if st.button("Mural", key="btn_mural"):
                st.switch_page("pages/comp_video.py")



