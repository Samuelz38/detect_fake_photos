import cv2
import numpy as np
import matplotlib.pyplot as plt

# Inicializar vídeo
cap = cv2.VideoCapture("teste.mp4")
ret, prev_frame = cap.read()
if not ret:
    raise ValueError("Não foi possível ler o vídeo")

cv2.split

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
# Preparar listas para armazenar métricas
mean_mags = []
std_mags = []


# Configurar matplotlib
plt.ion()
fig, ax = plt.subplots(2, 1, figsize=(8,6))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    # Métricas
    mean_mag = np.mean(mag)
    std_mag  = np.std(mag)
    mean_mags.append(mean_mag)
    std_mags.append(std_mag)

    # Mapa de calor do movimento
    hsv = np.zeros_like(frame)
    hsv[...,1] = 255
    hsv[...,0] = ang * 180 / np.pi / 2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Mostrar vídeo e mapa de fluxo
    cv2.imshow("Frame", frame)
    cv2.imshow("Optical Flow Heatmap", flow_rgb)

    # Atualizar gráficos
    ax[0].cla()
    ax[0].plot(mean_mags, label='Média da magnitude')
    ax[0].legend()
    ax[0].set_ylabel("Magnitude")
    ax[0].set_title("Média do movimento ao longo do tempo")

    ax[1].cla()
    ax[1].plot(std_mags, label='Desvio padrão da magnitude', color='orange')
    ax[1].legend()
    ax[1].set_ylabel("Desvio Padrão")
    ax[1].set_xlabel("Frame")
    ax[1].set_title("Desvio padrão do movimento ao longo do tempo")

    plt.pause(0.001)

    prev_gray = gray


    if cv2.waitKey(20) & 0xFF == 27:  # ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
