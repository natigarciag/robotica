import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
import sys

def clasificar(shapeD):

	dataset = np.memmap('dataset.driver', dtype='uint8', mode='r', shape=shapeD)

	###Clasificador
	Xtrain = dataset[:,0:-1]
	ytrain = dataset[:,-1]

	dTot=Xtrain

	Xtest= Xtrain[-40000:]
	ytest = ytrain[-40000:]
	Xtrain = Xtrain[0:-40000]
	ytrain = ytrain[0:-40000]


	clf = NearestCentroid()
	clf.fit(Xtrain, ytrain)

	NearestCentroid(metric='euclidean', shrink_threshold=None)

	res = clf.predict(Xtest)

	tot = len(Xtest)
	aci = len(res[res==ytest])
	print("Datos totales")
	print(len(dTot))
	print(tot)
	print(aci)
	print(len(Xtest))
	print(100*(float(aci)/float(tot)))
