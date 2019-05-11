import cv2
from matplotlib import pyplot as plt
import numpy as np

# capture = cv2.VideoCapture('./videos/video.mp4')
capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# capture.set(cv2.CAP_PROP_SATURATION, 150)

# capture.set(cv2.CAP_PROP_APERTURE, 0)
# capture.set(cv2.CAP_PROP_EXPOSURE, 0.1)
# capture.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 0.2)
# capture.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, 0.8)
# capture.set(cv2.CAP_PROP_BRIGHTNESS, 120)
capture.set(cv2.CAP_PROP_CONTRAST, 200)


