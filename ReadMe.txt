- Use o comando 'streamlit run app.py' no seu cmd 
  para iniciar a aplicação.

- Em src/layout: temos o layout das páginas (aqui você pode modificar a estrutura da página e os processos)
  - home
  - results_img
  - video_comp

- Em src/ultils: temos as funções de processamento de imagem e vídeo com o opencv 
  - ultil.py

- pages: comtem a configuração de cada página (exceto home, que está em app.py) e o inicializador delas
  - comp_video.py
  - result_img.py

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

COMANDOS QUE VOCÊ PODE USAR PARA O PROCESSMENTO DE VIDEOS

obs: st.file_uploader retorna um objeto do tipo UploadedFile, mas o cv2.VideoCapture espera um caminho de arquivo (string) ou um dispositivo de captura                                   (número inteiro).

# Extração e Pré-processamento
- cv2.VideoCapture 
- cv2.resize 
- cv2.cvtColor 
- cv2.convertScaleAbs 
- cv2.inRange 
- cv2.bitwise_and 

# Detecção de bordas
- cv2.Canny
- cv2.Sobel

# Estatística e fluxo óptico
- cv2.meanStdDev
- cv2.medianBlur
- cv2.absdiff
- cv2.calcOpticalFlowFarneback
