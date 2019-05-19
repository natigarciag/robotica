import cv2
import time

# # capture = cv2.VideoCapture('./videos/video.mp4')
# capture = cv2.VideoCapture('./videos/circuitoSalaAlManzana1.mp4')
capture = cv2.VideoCapture('./videos/circuito_EDIT_EDIT.mp4')
# capture = cv2.VideoCapture('./videos/cruce4salidas.mp4')

# for i in range(1,20):
#     capture = cv2.VideoCapture(i)
#     status, im = capture.read()
#     if (status == True):
#         break

# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)







# No funciona eso de abajo, solo contrast
# time.sleep(2)
# capture.set(cv2.CAP_PROP_CONTRAST, 10)
# capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
# capture.set(cv2.CAP_PROP_EXPOSURE, 3)
# capture.set(cv2.CAP_PROP_BRIGHTNESS, 2)
# capture.set(cv2.CAP_PROP_SATURATION, 2)
# capture.set(cv2.CAP_PROP_CONTRAST, 200)

# Que no funcionaron/no ayudaron:
# capture.set(cv2.CAP_PROP_SATURATION, 150)
# capture.set(cv2.CAP_PROP_APERTURE, 0)
# capture.set(cv2.CAP_PROP_EXPOSURE, 0.1)
# capture.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 0.2)
# capture.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 0.8)
# capture.set(cv2.CAP_PROP_BRIGHTNESS, 120)
