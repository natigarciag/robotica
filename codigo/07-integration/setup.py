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
from joblib import dump, load

import cameraCapture as captureType
capture = captureType.capture

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

#GUARDAR MODELO JOBLIB
dump(symbolClassifier, './iconsModel.joblib',compress=True) 

#CARGAR MODELO JOBLIB
#symbolClassifier = load('./iconsModel.joblib') 

#GUARDAR MODELO JOBLIB
dump(segmenter, './segmentationModel.joblib',compress=True) 

#CARGAR MODELO JOBLIB
#segmenter = load('./segmentationModel.joblib') 


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


# outIm = cv2.VideoWriter('./videos/cruce4salidas.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, (320, 240))
# outSeg = cv2.VideoWriter('./videos/demoSegm.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, (320/4, 240/4))
