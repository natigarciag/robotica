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

loo = LeaveOneOut()
for train_index, test_index in loo.split(datos):
	#print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = datos[:,:-1][train_index], datos[:,:-1][test_index]
	y_train, y_test = datos[:,-1][train_index], datos[:,-1][test_index]
	#print(X_train, X_test, y_train, y_test)

	#print(X_train.shape)
	#print(X_test)
	distEuc = [np.linalg.norm(medDatos[i,:-1]-X_test) for i in range(len(medDatos))]
	#distEuc = np.dot(np.concatenate((X_train,np.ones((len(X_train[:,1]),1))),axis=1),medDatos[:,:-1].T)
	
	#print np.argmin(distEuc)
	#print(y_test)
	
	res.append([np.argmin(distEuc) == (y_test)]) 


print(np.sum(res)*100.0 / len(res))
