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


import sklearn.metrics as metrics
import sklearn.ensemble as ens

import sklearn as sk


# symbolClassifier = sk.tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=15)
symbolClassifier = KNeighborsClassifier(n_neighbors=1)
# symbolClassifier = sk.naive_bayes.GaussianNB()

def train(datos):
    trained = False
    while (not trained):
        X_train, y_train = datos[:,:-1], datos[:,-1]
        symbolClassifier.fit(X_train, y_train)

        prediction = symbolClassifier.predict(X_train)

        confusionMatrix = metrics.confusion_matrix(y_train, prediction)
        print(confusionMatrix)
        # sumConfMat = np.sum(confusionMatrix, axis=0)
        # if (sumConfMat[0] == 0 or sumConfMat[1] == 0 or sumConfMat[2] == 0 or sumConfMat[3] == 0):
        # 	print 'confusion matrix is confused. Will train again'
        # 	symbolClassifier.fit(X_train, y_train)
        # else:
        trained = True
