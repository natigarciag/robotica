# Autores:
# Luciano Garcia Giordano - 150245
# Gonzalo Florez Arias - 150048
# Salvador Gonzalez Gerpe - 150044


import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
import sklearn
from sklearn.model_selection import train_test_split
import sklearn.naive_bayes as nb
import sklearn.svm as svm
import sklearn.neural_network as nn
import sklearn.metrics as metrics
import sys
import config

class Clasificador():
	def __init__(self, shapeD):
		self.dataset = np.memmap('./datasets/dataset.driver', dtype='uint8', mode='r', shape=shapeD)

	def train(self):
		X = self.dataset[:,0:-1]
		Y = self.dataset[:,-1]

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.1, shuffle=True)
		
		# self.clf = NearestCentroid(metric='euclidean', shrink_threshold=None)
		self.clf = nb.GaussianNB()
		# self.clf = nn.MLPClassifier(hidden_layer_sizes=(4,3), activation='logistic', alpha=0.001)
		self.clf.fit(self.X_train, self.y_train)


		print 'Accuracy is: ' + str(self.clf.score(self.X_test, self.y_test))

		confusionMatrix = metrics.confusion_matrix(self.y_test, self.clf.predict(self.X_test))

		print 'Confusion matrix:'
		print confusionMatrix

	def predict(self,X):
		return self.clf.predict(X)

# def clasificar(shapeD):

# 	dataset = np.memmap('dataset.driver', dtype='uint8', mode='r', shape=shapeD)

# 	###Clasificador
	

# 	res = clf.predict(Xtest)

# 	tot = len(Xtest)
# 	aci = len(res[res==ytest])
# 	print("Datos totales")
# 	print(len(dTot))
# 	print(tot)
# 	print(aci)
# 	print(len(Xtest))
# 	print(100*(float(aci)/float(tot)))
