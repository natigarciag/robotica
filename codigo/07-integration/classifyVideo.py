# Autores:
# Luciano Garcia Giordano - 150245
# Gonzalo Florez Arias - 150048
# Salvador Gonzalez Gerpe - 150044
 
import numpy as np
import cv2
from matplotlib import pyplot as plt
import config
import datasetGenerator
import clasificador
import math
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.neighbors import DistanceMetric
from scipy.spatial import distance
import sklearn.neural_network as nn
import sklearn as sk
import sklearn.metrics as metrics
import sklearn.ensemble as ens

showRawImage = True
rawImageShrinkFactor = 1
showSegmentedImage = True
segmentedImageShrinkFactor = 1
showArrowSegmentation = True
arrowSegmentationShrinkFactor = 1
showLineSegmentation = False
lineSegmentationShrinkFactor = 1

numClasses = 4

datosCaballero = np.loadtxt('data_caballero.txt', delimiter=" ")
datosFlecha = np.loadtxt('data_flecha.txt', delimiter=" ")
datosCruz = np.loadtxt('data_cruz.txt', delimiter=" ")
datosCabina = np.loadtxt('data_cabina.txt', delimiter=" ")
datosEscalera = np.loadtxt('data_escalera.txt', delimiter=" ")


datosCaballero = np.c_[ datosCaballero, np.zeros(len(datosCaballero)) ] 
datosEscalera = np.c_[ datosEscalera, np.ones(len(datosEscalera)) ] 
datosCruz = np.c_[ datosCruz, np.full((len(datosCruz),1),2) ] 
datosCabina = np.c_[ datosCabina, np.full((len(datosCabina),1),3) ] 
datosFlecha = np.c_[ datosFlecha, np.full((len(datosFlecha),1),4) ] 


datos = np.concatenate((datosCaballero,datosCruz),axis=0)
datos = np.concatenate((datos,datosCruz),axis=0)
datos = np.concatenate((datos,datosCabina),axis=0)
datos = np.concatenate((datos,datosEscalera),axis=0)
datos = np.concatenate((datos,datosFlecha),axis=0)

namesOfTheShapes = ['servicio de caballero', 'escalera', 'cruz', 'cabina', 'flecha']

np.random.shuffle(datos)
res = []

import clasificadorFormas
clasificadorFormas.train(datos)
symbolClassifier = clasificadorFormas.symbolClassifier

segmenter = clasificador.Clasificador(datasetGenerator.shapeD)
segmenter.train()


# capture = cv2.VideoCapture('./videos/circuitoSalaAlManzana1.mp4')
capture = cv2.VideoCapture('./videos/circuito_EDIT_EDIT.mp4')
paleta = np.array([[0,0,255],[0,255,0],[255,0,0], [0,0,0]],dtype='uint8')  

shrinkFactor = 1
originalImageHeight = (config.imageShape['height'] / shrinkFactor)
imageHeight = int(originalImageHeight*0.7)
imageWidth = int(config.imageShape['width'] / shrinkFactor)

segImg = np.empty((imageHeight,
                   imageWidth),
                  dtype='uint8')

def touchingEdges(segmentation, threshold):
    if (np.sum(segmentation[0]) > threshold or np.sum(segmentation[segmentation.shape[0]-1]) > threshold or np.sum(segmentation[:,0]) > threshold or np.sum(segmentation[:,segmentation.shape[1]-1]) > threshold):
        return True
    else:
        return False


times = []
try:
    while (capture.isOpened()):
        beg = time.time()
        
        ret, im = capture.read()
        if showRawImage:
            cv2.imshow('raw image', im[0::rawImageShrinkFactor,0::rawImageShrinkFactor,:])

        # prepare segmentation
        imHSV = im[((originalImageHeight - imageHeight)*shrinkFactor)::shrinkFactor, 0::shrinkFactor, :]
        imHSV = cv2.cvtColor(imHSV, cv2.COLOR_BGR2HSV)
        imHS = imHSV[:,:,(0,1)]
        
        # perform segmentation
        def predictRow(i):
            segImg[i] = segmenter.predict(imHS[i])
        [predictRow(i) for i in range(imHS.shape[0])]

        # separate symbols from lines
        arrow = np.zeros(segImg.shape, dtype='uint8')
        line = np.zeros(segImg.shape, dtype='uint8')
        # 0 - line
        # 1 - floor
        # 2 - symbols
        # 3 - nothing - not used
        line[segImg == 0] = 1
        arrow[segImg == 2] = 1
        arrow = cv2.erode(arrow, None, dst=arrow, iterations=1)
        
        # if showSegmentedImage:
        #     cv2.imshow('segmented image', cv2.cvtColor(paleta[segImg], cv2.COLOR_RGB2BGR))
        showImg = np.copy(segImg)
        showImg[showImg == 2] = 1
        showImg[arrow == 1] = 2
        if showSegmentedImage:
            cv2.imshow('segmented treated image', cv2.cvtColor(paleta[showImg][0::segmentedImageShrinkFactor,0::segmentedImageShrinkFactor,:], cv2.COLOR_RGB2BGR))

        if showArrowSegmentation:
            cv2.imshow('arrows', (arrow*255)[0::arrowSegmentationShrinkFactor,0::arrowSegmentationShrinkFactor])
        if showLineSegmentation:
            cv2.imshow('line', (line*255)[0::lineSegmentationShrinkFactor,0::lineSegmentationShrinkFactor])

        touchesEdges = touchingEdges(arrow, 1)

        if(np.sum(arrow) > int(200 / (shrinkFactor*shrinkFactor))):
            if (not touchesEdges):
                moments = cv2.HuMoments(cv2.moments(arrow)).flatten()
                # print(moments)
                predictedShape = symbolClassifier.predict(np.array([moments]))
                # print(predictedShape, namesOfTheShapes[int(predictedShape[0])])
                # print(predictedShape)
                print(namesOfTheShapes[int(predictedShape[0])])
            else:
                print('touches edges')
        else:
            print('nothing')

        times.append(time.time() - beg)

        cv2.waitKey(1)
    
except TypeError as a:
    print(a)
finally:
    # print(times)
    print(np.mean(np.array(times)), 'was the time per frame')