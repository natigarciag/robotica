import cv2
import numpy as np

points = [
    [10,10],
    [20,20],
    [30,30],
    [40,40],
    [50,50],
    # [50,60]
]

print cv2.fitEllipse(np.array(points)*2)