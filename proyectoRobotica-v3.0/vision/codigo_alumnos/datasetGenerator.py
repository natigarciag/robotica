import pygame
import numpy as np
import cv2
from sklearn.neighbors.nearest_centroid import NearestCentroid

numberOfImages = 4

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


###Clasificador
Xtrain = hsExpanded[hsExpanded[:,2] != 0][:,0:-1]
ytrain = hsExpanded[hsExpanded[:,2] != 0][:,-1]

Xtest= Xtrain[-20000:]
ytest = ytrain[-20000:]
Xtrain = Xtrain[0:-20000]
ytrain = ytrain[0:-20000]


clf = NearestCentroid()
clf.fit(Xtrain, ytrain)

NearestCentroid(metric='euclidean', shrink_threshold=None)

res = clf.predict(Xtest)

tot = len(Xtest)
aci = len(res[res==ytest])
print(tot)
print(aci)
print(len(Xtest))
print(100*(float(aci)/float(tot)))

# data_marca=hsVector[np.where(np.all(np.equal(markedImage,(255,0,0)),2))]
# data_fondo=hsVector[np.where(np.all(np.equal(markedImage,(0,255,0)),2))]
# data_linea=hsVector[np.where(np.all(np.equal(markedImage,(0,0,255)),2))]

# print(data_marca)
