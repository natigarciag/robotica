import pygame
import numpy as np
import cv2
from sklearn.neighbors.nearest_centroid import NearestCentroid
import clasificadorEuc

numberOfImages = 6

hsImages = np.memmap('hsImages.driver', dtype='uint8', mode='r', shape=(numberOfImages, 240, 320, 2))
markedImages = np.memmap('markedImages.driver', dtype='uint8', mode='r', shape=(numberOfImages, 240, 320, 3))

hsVector = hsImages.reshape((numberOfImages*240*320,2))
# hsVector = hsImages
markedVector = markedImages.reshape((numberOfImages*240*320,3))
# markedVector = markedImages
# print(markedVector[:,0,0])

# print(markedImages[markedVector == np.array([0,0,0], dtype='float32')[np.newaxis, np.newaxis]])

# markedImage = markedVector[0]

hsExpanded = np.zeros((numberOfImages*240*320,3))
hsExpanded[:,:-1] = hsVector

for i in range(hsExpanded.shape[0]):
    pixel = markedVector[i]
    if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 255: # linea
        pixelClass = 1
    elif pixel[0] == 0 and pixel[1] == 255 and pixel[2] == 0: # suelo
        pixelClass = 2
    elif pixel[0] == 255 and pixel[1] == 0 and pixel[2] == 0: # marca
        pixelClass = 3
    else:
        pixelClass = 0
        
    hsExpanded[i,2] = pixelClass

#print(hsExpanded[hsExpanded[:,2] != 0])


shapeD = hsExpanded[hsExpanded[:,2] != 0].shape
dataset = np.memmap('dataset.driver', dtype='uint8', mode='w+', shape=shapeD)

dataset[:] = hsExpanded[hsExpanded[:,2] != 0]
dataset.flush()

clasificadorEuc.clasificar(shapeD)



