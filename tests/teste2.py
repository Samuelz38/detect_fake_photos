import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configurações
PATH_VIDEO = 'teste.mp4'



cap = cv2.VideoCapture(PATH_VIDEO)
ret, frame = cap.read()
gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gray_frame, gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("Frame", frame)
    cv2.imshow("Optical Flow", flow_rgb)

    prev_gray = gray

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()