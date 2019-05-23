import numpy as np
import cv2
from matplotlib import pyplot as plt
import config
# import datasetGenerator
import clasificador
import math
import os
import time
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import LeaveOneOut
# from sklearn.model_selection import KFold
from sklearn.neighbors import DistanceMetric
from scipy.spatial import distance
import sklearn.neural_network as nn
import sklearn as sk
import sklearn.metrics as metrics
import sklearn.ensemble as ens
from joblib import dump, load
import datetime

# 0 - line
# 1 - floor
# 2 - symbols
# 3 - nothing - not used


showRawImage = False
rawImageShrinkFactor = 1
showSegmentedImage = True
segmentedImageShrinkFactor = 1
showArrowSegmentation = False
arrowSegmentationShrinkFactor = 4
showLineSegmentation = False
lineSegmentationShrinkFactor = 1
drawAndRecordSchematicSegmentation = True


import cameraCapture as captureType
capture = captureType.capture

# Load segmenter
# import datasetGenerator
# segmenter = clasificador.Clasificador(datasetGenerator.shapeD)
# segmenter.train()
# dump(segmenter, './segmentationModel0ContrastAlpera.joblib',compress=True)
segmenter = load('./segmentationModel0ContrastAlpera.joblib')

paleta = np.array([[0,0,255],[0,255,0],[255,0,0], [0,0,0]],dtype='uint8')  

shrinkFactor = 1 # 2 funciona relativamente bien
originalImageHeight = config.imageShape['height']
imageHeight = int(originalImageHeight)
imageWidth = config.imageShape['width']

segImg = np.empty((imageHeight // shrinkFactor,
                   imageWidth // shrinkFactor),
                  dtype='uint8')

if not os.path.exists('./capturedVideos'):
    os.makedirs('./capturedVideos')

if drawAndRecordSchematicSegmentation:
    date = datetime.datetime.now()
    schematicsVideoOutput = cv2.VideoWriter('./capturedVideos/schematicsCapture_' + date.strftime("%m_%d_%H_%M") + '.mp4', cv2.VideoWriter_fourcc(*'XVID'), 5, (imageWidth//shrinkFactor, imageHeight//shrinkFactor))
    print schematicsVideoOutput

rawVideoOutput = cv2.VideoWriter('./capturedVideos/rawCapture_' + date.strftime("%m_%d_%H_%M") + '.mp4', cv2.VideoWriter_fourcc(*'XVID'), 15, (config.imageShape['width'], config.imageShape['height']))