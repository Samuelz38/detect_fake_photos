import cv2
import numpy as np
from datetime import datetime



def covert_to_rgb(img):
    file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image



def process_image(img, option, K=10):
    match option:
        case 'k-means':
            img = cv2.resize(img, (800, 600))
            Z = img.reshape((-1, 3))
            Z = np.float32(Z)
            
            # Critérios de parada
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

            # K-Means
            ret_kmeans, label, center = cv2.kmeans(
                Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )
            center = np.uint8(center)
            res = center[label.flatten()]
            img_kmeans = res.reshape((img.shape))

            return img_kmeans

