import streamlit as st
from ..utils.util import process_video

def render_result_page_video():
    st.set_page_config(
        page_title="Análise de Vídeo",
        page_icon="🎬",
        layout="wide"
    )

    st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #1f77b4; margin-bottom: 30px}
    .result-section {padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin: 10px 0}
    .metric-box {padding: 15px; border-radius: 8px; background-color: black; margin: 5px}
    .analysis-mode {background-color: #e6f7ff; padding: 15px; border-radius: 10px; margin-bottom: 20px}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-header">🔍 Análise Forense de Vídeo</p>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("🔧 Modo de Análise")
        analysis_mode = st.radio(
            "Selecione o tipo de análise:",
            ["Comparação entre dois vídeos", "Análise individual"],
            help="Escolha entre comparar dois vídeos ou analisar um vídeo individualmente"
        )
        
        st.header("📁 Upload de Arquivos")
        
        if analysis_mode == "Comparação entre dois vídeos":
            upload_file = st.file_uploader("Vídeo Original", type=['mp4'], help="Carregue o vídeo original para análise")
            upload_file_fake = st.file_uploader("Vídeo Suspeito", type=['mp4'], help="Carregue o vídeo potencialmente manipulado")
        else:
            upload_file = st.file_uploader("Vídeo para Análise", type=['mp4'], help="Carregue o vídeo para análise")
            upload_file_fake = None
        
        st.markdown("---")
        st.header("⚙️ Configurações")
        option = st.selectbox("Método de Análise:", [
            "Análise de Metadados",
            "Bordas (Canny)",
            "Análise de Inconsistências de Compressão",
            "Análise de Padrões de Ruído",
            "Optical Analise"
        ], help="Selecione o método de análise forense")

    # Verificar se há arquivos para processar
    if analysis_mode == "Comparação entre dois vídeos":
        files_ready = upload_file is not None and upload_file_fake is not None
    else:
        files_ready = upload_file is not None

    if files_ready:
        if analysis_mode == "Comparação entre dois vídeos":
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎬 Vídeo Original")
                st.video(upload_file)
                
            with col2:
                st.subheader("🔍 Vídeo Suspeito")
                st.video(upload_file_fake)
        else:
            st.subheader("🎬 Vídeo para Análise")
            st.video(upload_file)

        if st.button("🚀 Executar Análise", type="primary", use_container_width=True):
            with st.spinner(f"Processando {option}..."):
                if analysis_mode == "Comparação entre dois vídeos":
                    processed_vid_ori = process_video(upload_file, option)
                    process_vid_fake = process_video(upload_file_fake, option)
                else:
                    processed_vid_ori = process_video(upload_file, option)
                    process_vid_fake = None

            st.success("✅ Análise concluída!")
            st.markdown("---")

            st.subheader("📊 Resultados da Análise")

            if option == 'Análise de Metadados':
                if analysis_mode == "Comparação entre dois vídeos":
                    col_meta1, col_meta2 = st.columns(2)
                    with col_meta1:
                        st.markdown("**Vídeo Original**")
                        frame, fps, frame_count, width, height, codec = processed_vid_ori
                        st.metric("FPS", f"{int(fps)}")
                        st.metric("Total de Frames", frame_count)
                        st.metric("Resolução", f"{width}x{height}")
                        st.metric("Codec", codec)
                        
                    with col_meta2:
                        st.markdown("**Vídeo Suspeito**")
                        frame_fake, fps_fake, frame_count_fake, width_fake, height_fake, codec_fake = process_vid_fake
                        st.metric("FPS", f"{int(fps_fake)}", delta=f"{int(fps_fake)-int(fps)}")
                        st.metric("Total de Frames", frame_count_fake, delta=f"{frame_count_fake-frame_count}")
                        st.metric("Resolução", f"{width_fake}x{height_fake}")
                        st.metric("Codec", codec_fake)
                else:
                    st.markdown("**Metadados do Vídeo**")
                    frame, fps, frame_count, width, height, codec = processed_vid_ori
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("FPS", f"{int(fps)}")
                    with col2:
                        st.metric("Total de Frames", frame_count)
                    with col3:
                        st.metric("Resolução", f"{width}x{height}")
                    with col4:
                        st.metric("Codec", codec)

            else:
                if analysis_mode == "Comparação entre dois vídeos":
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.markdown("**Vídeo Original**")
                        st.markdown(f'<div class="metric-box">{processed_vid_ori}</div>', unsafe_allow_html=True)
                    
                    with res_col2:
                        st.markdown("**Vídeo Suspeito**")
                        st.markdown(f'<div class="metric-box">{process_vid_fake}</div>', unsafe_allow_html=True)

                    if isinstance(processed_vid_ori, (int, float)):
                        st.markdown("---")
                        st.subheader("📈 Comparação Visual")
                        chart_data = {
                            "Tipo": ["Original", "Suspeito"],
                            "Valor": [processed_vid_ori, process_vid_fake]
                        }
                        st.bar_chart(chart_data, x="Tipo", y="Valor")
                
                    else:
                        pass    
                
                else:
                    st.markdown("**Resultado da Análise**")
                    st.markdown(f'<div class="metric-box">{processed_vid_ori}</div>', unsafe_allow_html=True)
                    
                    if isinstance(processed_vid_ori, (int, float)):
                        st.markdown("---")
                        st.subheader("📊 Valor Obtido")
                        st.metric("Resultado", f"{processed_vid_ori:.4f}")

                    else:
                        pass
    
    else:
        if analysis_mode == "Comparação entre dois vídeos":
            st.info("👆 Faça upload de ambos os vídeos para iniciar a análise")
        else:
            st.info("👆 Faça upload de um vídeo para iniciar a análise")
        st.image("Error-Detection.png", use_container_width=True)

if __name__ == "__main__":

    render_result_page_video()
