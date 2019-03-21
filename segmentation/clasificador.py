# Autores:
# Luciano Garcia Giordano - 150245
# Gonzalo Florez Arias - 150048
# Salvador Gonzalez Gerpe - 150044


import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
import sklearn
#from sklearn.model_selection import train_test_split
import sklearn.naive_bayes as nb
import sklearn.svm as svm
import sklearn.neural_network as nn
import sklearn.metrics as metrics
import sys
import config

class Clasificador():
	def __init__(self, shapeD):
		self.dataset = np.memmap('./datasets/datasetManzana.driver', dtype='uint8', mode='r', shape=shapeD)

	def train(self):
		indices = np.random.permutation(self.dataset.shape[0])
		cut = int(self.dataset.shape[0]*0.9)
		training_idx, test_idx = indices[:cut], indices[cut:]
		train, test = self.dataset[training_idx,:], self.dataset[test_idx,:]

		self.X_train = train[:,0:-1]
		self.X_test = test[:,0:-1]
		self.y_train = train[:,-1]
		self.y_test = test[:,-1]

		# self.clf = NearestCentroid(metric='euclidean', shrink_threshold=None)
		# self.clf = nb.GaussianNB()
		self.clf = nn.MLPClassifier(hidden_layer_sizes=(3), activation='logistic', alpha=0.005, momentum=0.1, warm_start=True)
		self.clf.fit(self.X_train, self.y_train)


		print 'Accuracy is: ' + str(self.clf.score(self.X_test, self.y_test))

		confusionMatrix = metrics.confusion_matrix(self.y_test, self.clf.predict(self.X_test))

		print 'Confusion matrix:'
		print confusionMatrix

	def predict(self,X):
		return self.clf.predict(X)