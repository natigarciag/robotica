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
import imutils

import time



clf = clasificadorEuc.Clasificador(datasetGenerator.shapeD)
clf.train()

# Inicio la captura de imagenes
# capture = cv2.VideoCapture('./videos/videoBola.mp4')
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Ahora clasifico el video
frame = 0

size = 1
while (True):
    # capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    # frame += 1 #2
    ret, im = capture.read()

    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    cv2.imshow('Real', im)

    # segmentation
    blurred = cv2.GaussianBlur(im, (7,7),0)
    imHSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    imHS = imHSV[:,:,(0,1)]

    paleta = np.array([[0,0,0],[0,0,255],[0,255,0],[255,0,0]],dtype='uint8')  

    # print(imHS[1,2])

    # segImg = np.array([[paleta[clf.predict(imHS[i,j])] for j in range(imHS.shape[1])] for i in range(imHS.shape[0])], dtype='uint8')
    segImg = np.array([paleta[clf.predict(imHS[i])] for i in range(imHS.shape[0])], dtype='uint8')

    whereIsObject = np.all(segImg == [0,255,0], axis=-1)
    whereIsObjectPositions = np.where(whereIsObject==True)
    
    if whereIsObjectPositions[1].shape[0] != 0:
        minx = np.min(whereIsObjectPositions[1])
      	maxx = np.max(whereIsObjectPositions[1])
        size = maxx - minx
        if size == 0:
            size = 1

    # dist = 37.0 # put fixed distance to the object here
    diameter = 12.0 # put size of object here
    # print dist*size/diameter #uncomment to see your value of paramF
    paramF = 292.9

    dist = paramF*diameter / size
    print dist


    cv2.imshow("Segmentacion Euclid",cv2.cvtColor(segImg,cv2.COLOR_RGB2BGR))
 
    cv2.waitKey(1)