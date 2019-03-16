# Autores:
# Luciano Garcia Giordano - 150245
# Gonzalo Florez Arias - 150048
# Salvador Gonzalez Gerpe - 150044

import cv2
from matplotlib import pyplot as plt
import numpy as np
import config
import datasetGenerator
import clasificadorEuc

import time

from joblib import Parallel, delayed



clf = clasificadorEuc.Clasificador(datasetGenerator.shapeD)
clf.train()

# Inicio la captura de imagenes
capture = cv2.VideoCapture('./videos/video.mp4')

# Ahora clasifico el video
frame = 0

paleta = np.array([[0,0,0],[0,0,255],[0,255,0],[255,0,0]],dtype='uint8')  

shrinkFactor = 4
segImg = np.empty((config.imageShape['height']/shrinkFactor, config.imageShape['width']/shrinkFactor, 3), dtype='uint8')

while (capture.isOpened()):
    times = []
    beg = time.time()
    
    # capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    # frame += 3
    ret, im = capture.read()
    
    times.append(time.time())

    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    cv2.imshow('Real', im)

    times.append(time.time())

    # im = cv2.GaussianBlur(im, (7,7), 0)

    times.append(time.time())

    # segmentation
    imHSV = im[0::shrinkFactor,0::shrinkFactor,:]
    times.append(time.time())
    imHSV = cv2.cvtColor(imHSV, cv2.COLOR_BGR2HSV)
    times.append(time.time())
    imHS = imHSV[:,:,(0,1)]
    times.append(time.time())

    # segImg = np.array([[paleta[clf.predict(imHS[i,j])] for j in range(imHS.shape[1])] for i in range(imHS.shape[0])], dtype='uint8')
    def predictRow(i):
        segImg[i] = paleta[clf.predict(imHS[i])]
        # if i > 238:
        #     times.append(time.time())


    # Parallel(n_jobs=4)(delayed(predictRow)(i) for i in range(imHS.shape[0]))
    times.append(time.time())
    [predictRow(i) for i in range(imHS.shape[0])]
    times.append(time.time())
    # end3 = time.time()

    # segImg = np.array([paleta[clf.predict(imHS[i])] for i in range(imHS.shape[0])], dtype='uint8')
    # imHS = np.reshape(imHS, ((config.imageShape['height']/2)*(config.imageShape['width']/2),2))
    # segImg = paleta[clf.predict(imHS)]
    # segImg = np.reshape(segImg, (config.imageShape['height']/2,config.imageShape['width']/2,3))

    times.append(time.time())
    cv2.imshow("Segmentacion Euclid",cv2.cvtColor(segImg,cv2.COLOR_RGB2BGR))
    times.append(time.time())

    differences = []
    for time1 in times:
        differences.append(time1 - beg)

    print differences
 
    cv2.waitKey(1)

# cv2.imshow("Segmentacion Euclid",cv2.cvtColor(segImg,cv2.COLOR_RGB2BGR))