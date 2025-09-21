import cv2
import numpy as np
import matplotlib.pylab as plt


def display_cv2_img(img,figsize=(10,10)):
    img_ = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_)
    ax.axis('off')




# Abrindo o Video e Capturando Metadados

cap = cv2.VideoCapture('teste.mp4')

FRAME_COUNT = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))
WIDTH = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
HEIGHT = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


# Extraindo imagens de vídeo

ret, frame = cap.read()
print(f'Returned {ret} e tamnho da imagem {frame.shape}')
display_cv2_img(frame)

# Exibir varios frames do video

fig, axs = plt.subplots(5, 5, figsize=(30,20))
axs = axs.flatten()
img_idx = 0

for frame in range(FRAME_COUNT):
    ret, img = cap.read()
    if ret == False:
        break
    if frame % 100 == 0:
        axs[img_idx].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        axs[img_idx].set_title(f'Frame:{frame}')
        axs[img_idx].axis('off')
        img_idx += 1

plt.tight_layout()
plt.show()


