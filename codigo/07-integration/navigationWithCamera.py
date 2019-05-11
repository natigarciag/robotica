# Autores:
# Luciano Garcia Giordano - 150245
# Gonzalo Florez Arias - 150048
# Salvador Gonzalez Gerpe - 150044

import cv2
from matplotlib import pyplot as plt
import numpy as np
import config
import datasetGenerator
import clasificador
import math

import time
import setup

def getSalidas(line):
    height = len(line) - 1  #240
    width = len(line[0]) - 1  #320

    line[0] = np.bitwise_and(line[0], line[1])
    line[height] = np.bitwise_and(line[height], line[height - 1])

    line[:,0] = np.bitwise_and(line[:,0],line[:,1])
    line[:,width] = np.bitwise_and(line[:,width], line[:,width - 1])

    InsOuts = []

    Sup = []
    i = 0

    for index, pixel in enumerate(line[0]):  #Fila Superior
        if pixel == 0 and i != 0:
            Sup.append([index - i, index - 1])
            i = 0
        elif pixel == 1:
            i = i + 1

    if pixel == 1:
        InsOuts.append([0, width])

    Inf = []
    i = 0

    for index, pixel in enumerate(line[height]):  #Fila Inferior
        if pixel == 0 and i != 0:
            Inf.append([index - i, index - 1])
            i = 0
        elif pixel == 1:
            i = i + 1

    if pixel == 1:
        InsOuts.append([height, width])

    Izq = []
    i = 0

    for index, pixel in enumerate([row[0] for row in line]):  #Fila Inferior
        if pixel == 0 and i != 0:
            Izq.append([index - i, index - 1])
            i = 0
        elif pixel == 1:
            i = i + 1

    if pixel == 1:
        InsOuts.append([height, 0])

    Der = []
    i = 0

    for index, pixel in enumerate(
        [row[width] for row in line]):  #Fila Inferior
        if pixel == 0 and i != 0:
            Der.append([index - i, index - 1])
            i = 0
        elif pixel == 1:
            i = i + 1

    if pixel == 1:
        InsOuts.append([height, width])


    for el in Izq:
        InsOuts.append([(el[1] - el[0]) / 2 + el[0], 0])
    for el in Sup:
        InsOuts.append([0, (el[1] - el[0]) / 2 + el[0]])
    for el in Der:
        InsOuts.append([(el[1] - el[0]) / 2 + el[0], width])
    for el in Inf:
        InsOuts.append([height, (el[1] - el[0]) / 2 + el[0]])


    Result = []
    for el in InsOuts:
        if el not in Result:
            Result.append(el)

    print 'Pixels', Result
    return Result

def getArrowPosition(arrow):
    if (np.sum(arrow) == 0):
        return (0,0),-1,[0,0],[[0,0]]
    positionsOfArrow = np.where(arrow == 1)
    # print 'Positions are', positionsOfArrow[0].shape
    if (positionsOfArrow[0].shape[0] < 5):
        return (0,0),-1,[0,0],[[0,0]]
    positionsOfArrow = np.dstack((positionsOfArrow[0], positionsOfArrow[1]))
    rectMin = cv2.minAreaRect(positionsOfArrow)
    rect = cv2.boxPoints(rectMin)

    if (np.min(rect[:,0]) < 0):
        return (0,0),-1,[0,0],[[0,0]]
    if (np.max(rect[:,0]) > imageHeight):
        return (0,0),-1,[0,0],[[0,0]]
    if (np.min(rect[:,1]) < 0):
        return (0,0),-1,[0,0],[[0,0]]
    if (np.max(rect[:,1]) > imageWidth):
        return (0,0),-1,[0,0],[[0,0]]
    
    if (rectMin[1][0]*rectMin[1][1] < 2):
        return (0,0),-1,[0,0],[[0,0]]
    geometricCenter = np.mean(rect, axis=0)
    if (geometricCenter[0] > 0.9 * (imageHeight)):
        return (0,0),-1,[0,0],[[0,0]]

    # Barycenter
    barycenter = np.mean(positionsOfArrow, axis=1)
    geometricCenterBarycenterVector = barycenter - geometricCenter
    # print(geometricCenterBarycenterVector)

    # alternate = geometricCenterBarycenterVector[0][0]
    # geometricCenterBarycenterVector[0][0] = geometricCenterBarycenterVector[0][1]
    # geometricCenterBarycenterVector[0][1] = alternate

    return geometricCenter, geometricCenterBarycenterVector, geometricCenter, barycenter

def decideEntrance(inputsOutputs):
    if len(inputsOutputs) <= 1:
        return [], []
        
    inOuts = np.array(inputsOutputs)
    centerOfImage = np.array([imageHeight,int(imageWidth/2)])
    
    matrix = np.power(inOuts-centerOfImage,2)
    
    reduced = np.sum(matrix, axis = 1) # 1 para mantener el numero de salidas

    inPut = np.argmin(reduced)
    

    inPuted = inOuts[inPut]
    
    inOuts = np.delete(inOuts,inPut,axis=0)

    return inPuted, inOuts

def decideExit(outputs, centerOfArrow, vectorOfArrow, imageOnPaleta):
    # Consideramos outputs una lista de vectores en Numpy

    arrow = vectorOfArrow[0]

    # print('outputs:', outputs)
    outVectors = outputs - centerOfArrow
    print('vectorsToOutputs', outVectors)

    for vect in outVectors:
        cv2.arrowedLine(
            imageOnPaleta, 
            (int(centerOfArrow[1]), int(centerOfArrow[0])), 
            (int(centerOfArrow[1] + vect[1]), int(centerOfArrow[0] + vect[0])),
            (255,255,255),
            1)

    print('arrow:', arrow)

    # print(outVectors)
    print(np.linalg.norm(outVectors))
    print(np.linalg.norm(arrow))
    print('dot', np.dot(outVectors, arrow))
    print('dot normalizado', np.dot(outVectors, arrow) / (np.linalg.norm(outVectors) * np.linalg.norm(arrow)))
    outvectorsInv = outVectors * np.array([-1,1])
    arrowInv = arrow * np.array([-1,1])
    angles = np.arccos(1 - np.abs(np.dot(outvectorsInv, arrowInv) / (np.linalg.norm(outvectorsInv) * np.linalg.norm(arrowInv))))
    print('angles:', (angles * 180) / math.pi)

    out = outputs[np.argmin(angles)]
    return out

def chooseIndexOfMostCentralExit(exitsArray):
    # exitsArray is an np array of exits. Choose the most "central top" one

    centerOfImage = np.array([0,int(imageWidth/2)])
    
    matrix = np.power(exitsArray-centerOfImage,2)
    reduced = np.sum(matrix, axis = 1)
    indexOfOutput = np.argmin(reduced)
    
    return indexOfOutput

times = []

previousAngle = 0
previousDistance = 0

while (capture.isOpened()):
    beg = time.time()

    ret, im = capture.read()
    # outIm.write(im)

    try:
        cv2.imshow('Real', im)
    except Exception as e:
        break
    

    # segmentation
    imHSV = im[((originalImageHeight - imageHeight)*shrinkFactor)::shrinkFactor, 0::shrinkFactor, :]
    imHSV = cv2.cvtColor(imHSV, cv2.COLOR_BGR2HSV)
    imHS = imHSV[:, :, (0, 1)]

    def predictRow(i):
        # print segImg
        segImg[i] = clf.predict(imHS[i])

    [predictRow(i) for i in range(imHS.shape[0])]

    imageOnPaleta = paleta[segImg]

    arrow = np.zeros(segImg.shape, dtype='uint8')
    line = np.zeros(segImg.shape, dtype='uint8')
    # 0 - line
    # 1 - floor
    # 2 - symbols
    # 3 - nothing - not used
    line[segImg == 0] = 1
    arrow[segImg == 2] = 1

    arrow = cv2.erode(arrow, None, dst=arrow, iterations=1)

    cv2.imshow('arrows', (arrow*255))

    salidas = getSalidas(line)
    centerOfArrow, vectorOfArrow, pto1, pto2 = getArrowPosition(arrow)

    if (vectorOfArrow is not -1):
        cv2.arrowedLine(
            imageOnPaleta, 
            (int(centerOfArrow[1]), int(centerOfArrow[0])), 
            (int(vectorOfArrow[0][1]*20 + centerOfArrow[1]), int(vectorOfArrow[0][0]*20 + centerOfArrow[0])),
            (0,0,255),
            1)
        # print(pto1)
        # print(pto2)
        imageOnPaleta[int(pto1[0]), int(pto1[1]),0] = 255
        imageOnPaleta[int(pto1[0]), int(pto1[1]),1] = 255
        imageOnPaleta[int(pto1[0]), int(pto1[1]),2] = 255
        imageOnPaleta[int(pto2[0][0]), int(pto2[0][1]),0] = 255
        imageOnPaleta[int(pto2[0][0]), int(pto2[0][1]),1] = 255
        imageOnPaleta[int(pto2[0][0]), int(pto2[0][1]),2] = 255
        
    entrance, exits = decideEntrance(salidas)

    if (len(entrance) > 0):
        # Since there's an entrance, show it on the image
        cv2.ellipse(imageOnPaleta,(int(entrance[1]), int(entrance[0])),(3,3),0,0,360,(0,255,255),-1)

        if (len(exits) > 0):
            for uniqueExit in exits:
                cv2.ellipse(imageOnPaleta,(uniqueExit[1], uniqueExit[0]),(3,3),0,0,360,(255,0,255),-1)
            exitsArray = np.array(exits)
            if vectorOfArrow is not -1 and len(exits) > 1:
                # When there's an arrow and many exits, decide from the arrow
                selectedExit = decideExit(exitsArray, centerOfArrow, vectorOfArrow, imageOnPaleta)
            else:
                # When there are many exits and no arrows, choose the most central one
                selectedExit = exits[chooseIndexOfMostCentralExit(exitsArray)]
                
            cv2.ellipse(imageOnPaleta,(selectedExit[1], selectedExit[0]),(3,3),0,0,360,(0,120,120),-1)
            cv2.arrowedLine(
                imageOnPaleta, 
                (int(entrance[1]), int(entrance[0])), 
                (int(selectedExit[1]), int(selectedExit[0])),
                (255,255,0),
                1)

            consignaArray = np.array([selectedExit[0] - entrance[0], selectedExit[1] - entrance[1]])
            distanceToEntrance = entrance[1] - int(imageWidth/2)
            verticalVector = np.array([0, -1])
            consignaAngle = np.arccos(np.dot(consignaArray, verticalVector) / (np.linalg.norm(consignaArray) * np.linalg.norm(verticalVector)))
            if (consignaArray[0] < 0):
                consignaAngle = -consignaAngle
            consignaAngle = (consignaAngle * 180) / np.pi

            # Calculating the consigna
            rotationD = previousAngle - (consignaAngle*0.7 + distanceToEntrance*0.3)
            rotationKd = 0.2
            rotation = (consignaAngle*0.03 + distanceToEntrance*0.01)/(math.fabs(rotationD)*rotationKd)
            speed = -0.5*math.fabs(rotation) + 1

            if (rotation > 1):
                rotation = 1
            elif (rotation < -1):
                rotation = -1

            if (speed > 1):
                speed = 1
            elif (speed < 0):
                speed = 0
            
            rotationRobot = -rotation

            print round(speed, 1), round(rotationRobot,1)

            previousAngle = consignaAngle
            previousDistance = distanceToEntrance

    end = time.time()
    times.append(end - beg)

    imgOnPaletaBGR = cv2.cvtColor(imageOnPaleta, cv2.COLOR_RGB2BGR)
    cv2.imshow("Segmentacion Euclid", imgOnPaletaBGR)

    cv2.waitKey(1)

    # outSeg.write(cv2.cvtColor(segImg,cv2.COLOR_RGB2BGR))

print 'mean time is', np.mean(np.array(times))
