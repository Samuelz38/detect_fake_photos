import streamlit as st
from datetime import datetime
from ..utils.util import covert_to_rgb, process_image

def render_result_page():
    st.title('Preparando Análise')
    
    with st.sidebar:
        st.header('Opções')
        upload_file = st.file_uploader('Carregue um arquivo JPEG',type=["jpg", "jpeg", "png"])

    if upload_file is not None:
        img = covert_to_rgb(upload_file)
        st.subheader("Imagem Original")
        st.image(img, use_container_width=True)

        option = st.selectbox("Escolha um processamento:", 
                          ["k-means", "Cinza", "Bordas (Canny)", "Blur"])

        if option == 'k-means':
            K = st.slider("Número de clusters (K)", min_value=2, max_value=20, value=10)

        with st.spinner(f"Processando imagem: {option}..."):
            processed_img = process_image(img, option, K)

        # Detecta se é grayscale para exibir corretamente
        if len(processed_img.shape) == 2:
            st.image(processed_img, use_container_width=True, channels="GRAY")
        else:
            st.image(processed_img, use_container_width=True)
        
        st.success("Processamento concluído!")

