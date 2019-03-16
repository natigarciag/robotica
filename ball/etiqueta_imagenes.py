# Autores:
# Luciano Garcia Giordano - 150245
# Gonzalo Florez Arias - 150048
# Salvador Gonzalez Gerpe - 150044

import cv2
# from scipy.misc import imread, imsave
from matplotlib import pyplot as plt
import select_pixels as sel
import numpy as np
import config


capture = cv2.VideoCapture('./videos/videoPelotaTenis.mp4')

hsImages = np.memmap('./datasets/hsImages.driver', dtype='uint8', mode='w+', shape=(config.numberOfImages, config.imageShape['height'], config.imageShape['width'], 2))
markedImages = np.memmap('./datasets/markedImages.driver', dtype='uint8', mode='w+', shape=(config.numberOfImages, config.imageShape['height'], config.imageShape['width'], 3))

for i in range(config.numberOfImages):
    print(i)

    # cImg = 0
    key = 0
    while (key != 27):
        ret, im = capture.read()

        cv2.imshow('Camera', im)
        
        key = cv2.waitKey(35)

    cv2.destroyWindow('Camera')
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    rgbImg = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    markedImg = sel.select_fg_bg(rgbImg)

    hsImages[i] = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2HSV)[:,:,(0,1)]
    markedImages[i] = markedImg

hsImages.flush()
markedImages.flush()
