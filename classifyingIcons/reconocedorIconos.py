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


datosCaballero = np.loadtxt('data_caballero.txt', delimiter=" ")
datosFlecha = np.loadtxt('data_flecha.txt', delimiter=" ")
datosCruz = np.loadtxt('data_cruz.txt', delimiter=" ")
datosCabina = np.loadtxt('data_cabina.txt', delimiter=" ")
datosEscalera = np.loadtxt('data_escalera.txt', delimiter=" ")




datosCaballero = np.c_[ datosCaballero, np.zeros(len(datosCaballero)) ] 
datosFlecha = np.c_[ datosFlecha, np.ones(len(datosFlecha)) ] 
datosCruz = np.c_[ datosCruz, np.full((len(datosCruz),1),2) ] 
datosCabina = np.c_[ datosCabina, np.full((len(datosCabina),1),3) ] 
datosEscalera = np.c_[ datosEscalera, np.full((len(datosEscalera),1),4) ] 

datos = np.concatenate((datosCaballero,datosFlecha),axis=0)
datos = np.concatenate((datos,datosCruz),axis=0)
datos = np.concatenate((datos,datosCabina),axis=0)
datos = np.concatenate((datos,datosEscalera),axis=0)


np.random.shuffle(datos)
datosTrain = datos[0:400]
datosTest = datos[400:]


neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(datosTrain[:,:-1], datosTrain[:,-1]) 


print("1NN:")
res = [neigh.predict(datosTest[:,:-1]) == datosTest[:,-1]]

print(np.sum(res)*100.0 / len(datosTest))


