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
from sklearn.neighbors import DistanceMetric
from scipy.spatial import distance


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

print("1NN:")

np.random.shuffle(datos)
datosTrain = datos[0:400]
datosTest = datos[400:]

res = []

loo = LeaveOneOut()
for train_index, test_index in loo.split(datos):
	#print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = datos[:,:-1][train_index], datos[:,:-1][test_index]
	y_train, y_test = datos[:,-1][train_index], datos[:,-1][test_index]
	#print(X_train, X_test, y_train, y_test)


	neigh = KNeighborsClassifier(n_neighbors=1)
	neigh.fit(X_train, y_train) 
	res.append([neigh.predict(X_test) == y_test]) 


print(np.sum(res)*100.0 / len(res))


print("Mahalanobis:")

res = []

medCaballero = np.mean(datosCaballero,0)
medFlecha = np.mean(datosFlecha,0)
medCruz = np.mean(datosCruz,0)
medCabina = np.mean(datosCabina,0)
medEscalera = np.mean(datosEscalera,0)

medDatos = np.c_[medCaballero,medFlecha].T
medDatos = np.concatenate((medDatos,np.array([medCruz])),axis=0)
medDatos = np.concatenate((medDatos,np.array([medCabina])),axis=0)
medDatos = np.concatenate((medDatos,np.array([medEscalera])),axis=0)


labels = np.unique(datos[:,-1])
X = datos[:,:-1]
y = datos[:,-1]

#print 
covDatos = np.array([ np.linalg.inv(np.cov(X[y == label])) for label in labels ]) 

print covDatos[0].shape



loo = LeaveOneOut()
for train_index, test_index in loo.split(medDatos):
	#print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = medDatos[:,:-1][train_index], medDatos[:,:-1][test_index]
	y_train, y_test = medDatos[:,-1][train_index], medDatos[:,-1][test_index]
	#print(X_train, X_test, y_train, y_test)

	#vi = np.linalg.inv(np.cov(X_train[0]))
	#print(np.cov(X_train).shape)
	print distance.mahalanobis(X_train[0],X_test,covDatos[0])
	
	

	neigh = KNeighborsClassifier(n_neighbors=1)
	neigh.fit(X_train, y_train) 
	res.append([neigh.predict(X_test) == y_test]) 


print(np.sum(res)*100.0 / len(res))


