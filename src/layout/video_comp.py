import streamlit as st
from datetime import datetime
from ..utils.util import process_video

def render_result_page_video():
    st.title('Preparando Análise')
    
    with st.sidebar:
        st.header('Opções')
        upload_file = st.file_uploader('Carregue um arquivo de vídeo original.',type=['mp4', 'mov', 'avi'])
        upload_file_fake = st.file_uploader('Carregue um arquivo de vídeo fake.',type=['mp4', 'mov', 'avi'])

    if upload_file is not None and upload_file_fake is not None:
        st.subheader("Vídeo Original")
        st.video(upload_file)

        option = st.selectbox("Escolha um processamento:", 
                          ["Análise de Metadados", "Bordas (Canny)", "Análise de Inconsistências de Compressão","Análise de Padrões de Ruído"])

    

        with st.spinner(f"Processando Vídeo: {option}..."):
            processed_vid_ori = process_video(upload_file,option)
            process_vid_fake = process_video(upload_file_fake,option)
        
        
        st.success("Processamento concluído!")

        if option == 'Análise de Metadados':
            frame,fps,frame_count,width,height,codec = processed_vid_ori
            frame_fake,fps_fake,frame_count_fake,width_fake,height_fake,codec_fake = process_vid_fake 
            st.write(f'Metadados Video Original\n- FPS: {int(fps)}\n- Frame_Count: {frame_count}\n- Largura: {width}\n- Altura: {height}\n- Codec: {codec} ')
            st.write(f'Metadados Video Falso\n- FPS: {int(fps_fake)}\n- Frame_Count: {frame_count_fake}\n- Largura: {width_fake}\n- Altura: {height_fake}\n- Codec: {codec_fake} ')
        
        if option == 'Análise de Padrões de Ruído':
            st.write('RESULTADO DA ANÁLISE:')
            st.write(f'- Original: {processed_vid_ori:.6f}')
            st.write(f'- Fake: {process_vid_fake:.6f}')
        
        if option == 'Análise de Inconsistências de Compressão':
            st.write('RESULTADO DA ANALISE:')
            st.write(f'- Original: {processed_vid_ori}')
            st.write(f'- Fake: {process_vid_fake}')
        
        if option == 'Bordas (Canny)':
            st.write('RESULTADO DA ANALISE:')
            st.write(f'- Original: {processed_vid_ori}')
            st.write(f'- Fake: {process_vid_fake}')
            