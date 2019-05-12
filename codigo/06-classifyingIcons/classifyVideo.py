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

numClasses = 4

datosCaballero = np.loadtxt('data_caballero.txt', delimiter=" ")
# datosFlecha = np.loadtxt('data_flecha.txt', delimiter=" ")
datosCruz = np.loadtxt('data_cruz.txt', delimiter=" ")
datosCabina = np.loadtxt('data_cabina.txt', delimiter=" ")
datosEscalera = np.loadtxt('data_escalera.txt', delimiter=" ")


datosCaballero = np.c_[ datosCaballero, np.zeros(len(datosCaballero)) ] 
datosEscalera = np.c_[ datosEscalera, np.ones(len(datosEscalera)) ] 
datosCruz = np.c_[ datosCruz, np.full((len(datosCruz),1),2) ] 
datosCabina = np.c_[ datosCabina, np.full((len(datosCabina),1),3) ] 
# datosFlecha = np.c_[ datosFlecha, np.full((len(datosFlecha),1),4) ] 


datos = np.concatenate((datosCaballero,datosCruz),axis=0)
datos = np.concatenate((datos,datosCruz),axis=0)
datos = np.concatenate((datos,datosCabina),axis=0)
datos = np.concatenate((datos,datosEscalera),axis=0)
# datos = np.concatenate((datos,datosFlecha),axis=0)

np.random.shuffle(datos)
res = []

# loo = KFold(n_splits=20, shuffle=True)
loo = LeaveOneOut()
for train_index, test_index in loo.split(datos):
	X_train, X_test = datos[:,:-1][train_index], datos[:,:-1][test_index]
	y_train, y_test = datos[:,-1][train_index], datos[:,-1][test_index]

	# print(X_train.shape)

	# classifier = nn.MLPClassifier(hidden_layer_sizes=(5), activation='logistic', alpha=0.08, momentum=0.1, verbose=True) #, early_stopping=True)
	# classifier = sk.naive_bayes.GaussianNB()
	# classifier = sk.svm.SVC(gamma='auto')
	# classifier = KNeighborsClassifier(n_neighbors=10)
	classifier = sk.tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=15)
	
	trained = False
	while (not trained):
		classifier.fit(X_train, y_train)

		prediction = classifier.predict(X_test)

		confusionMatrix = metrics.confusion_matrix(y_test, prediction)
		# print(confusionMatrix)
		# sumConfMat = np.sum(confusionMatrix, axis=0)
		# if (sumConfMat[0] == 0 or sumConfMat[1] == 0 or sumConfMat[2] == 0 or sumConfMat[3] == 0):
		# 	print 'confusion matrix is confused. Will train again'
		# 	classifier.fit(X_train, y_train)
		# else:
		trained = True
		res.append(metrics.accuracy_score(y_test, prediction))	



symbolClassifier = sk.tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=15)
# symbolClassifier = KNeighborsClassifier(n_neighbors=1)
# symbolClassifier = sk.naive_bayes.GaussianNB()
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
        # cv2.imshow('Real', im)

        # segmentation
        imHSV = im[((originalImageHeight - imageHeight)*shrinkFactor)::shrinkFactor, 0::shrinkFactor, :]
        # imHSV = im[0::shrinkFactor,0::shrinkFactor,:]
        # times.append(time.time())
        imHSV = cv2.cvtColor(imHSV, cv2.COLOR_BGR2HSV)
        # times.append(time.time())
        imHS = imHSV[:,:,(0,1)]
        # times.append(time.time())

        def predictRow(i):
            segImg[i] = segmenter.predict(imHS[i])

        [predictRow(i) for i in range(imHS.shape[0])]

        # imageOnPaleta = paleta[segImg]

        arrow = np.zeros(segImg.shape, dtype='uint8')
        line = np.zeros(segImg.shape, dtype='uint8')
        # 0 - line
        # 1 - floor
        # 2 - symbols
        # 3 - nothing - not used
        line[segImg == 0] = 1
        arrow[segImg == 2] = 1
        arrow = cv2.erode(arrow, None, dst=arrow, iterations=1)
        # arrow = cv2.dilate(arrow, None, dst=arrow, iterations=1)

        # print(np.sum(arrow))

        namesOfTheShapes = ['servicio de caballero', 'escalera', 'cruz', 'cabina', 'flecha']

        # cv2.imshow('name', cv2.cvtColor(paleta[segImg], cv2.COLOR_RGB2BGR))
        showImg = np.copy(segImg)
        showImg[showImg == 2] = 1
        showImg[arrow == 1] = 2
        cv2.imshow('name', cv2.cvtColor(paleta[showImg], cv2.COLOR_RGB2BGR))



        cv2.imshow('arrows', arrow*255)

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

        # times.append(time.time())
        # cv2.imshow("Segmentacion Euclid",cv2.cvtColor(segImg,cv2.COLOR_RGB2BGR))
        
        # times.append(time.time())

        times.append(time.time() - beg)
        
        # differences = []
        # for time1 in times:
        #     differences.append(time1 - beg)

        # print np.mean(np.array(times))
    
        cv2.waitKey(1)
    
except TypeError as a:
    pass
finally:
    # print(times)
    print(np.mean(np.array(times)), 'was the time per frame')
