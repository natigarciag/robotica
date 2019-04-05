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

import time

clf = clasificador.Clasificador(datasetGenerator.shapeD)
clf.train()

# Inicio la captura de imagenes
capture = cv2.VideoCapture('./videos/video.mp4')
# capture = cv2.VideoCapture(0)
# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# capture.set(cv2.CAP_PROP_SATURATION, 150)

# Ahora clasifico el video
frame = 0



paleta = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 0, 0]],
                  dtype='uint8')

shrinkFactor = 4
originalImageHeight = (config.imageShape['height'] / shrinkFactor)
imageHeight = int(originalImageHeight*0.5)
imageWidth = int(config.imageShape['width'] / shrinkFactor)

segImg = np.empty((imageHeight,
                   imageWidth),
                  dtype='uint8')

# outIm = cv2.VideoWriter('./videos/demoImagen.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, (320, 240))
# outSeg = cv2.VideoWriter('./videos/demoSegm.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, (320/4, 240/4))


def getSalidas(line):
    height = len(line) - 1  #240
    width = len(line[0]) - 1  #320

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

    for el in Sup:
        InsOuts.append([0, (el[1] - el[0]) / 2 + el[0]])
    for el in Inf:
        InsOuts.append([height, (el[1] - el[0]) / 2 + el[0]])
    for el in Izq:
        InsOuts.append([(el[1] - el[0]) / 2 + el[0], 0])
    for el in Der:
        InsOuts.append([(el[1] - el[0]) / 2 + el[0], width])

    Result = []
    for el in InsOuts:
        if el not in Result:
            Result.append(el)

    # print 'Pixels', Result
    return Result


def getArrowPosition(arrow):
    if (np.sum(arrow) == 0):
        return (0, 0), -1
    positionsOfArrow = np.where(arrow == 1)
    # print 'Positions are', positionsOfArrow[0].shape
    if (positionsOfArrow[0].shape[0] < 5):
        return (0, 0), -1
    positionsOfArrow = np.dstack((positionsOfArrow[0], positionsOfArrow[1]))
    # print positionsOfArrow.shape
    # geometricCenter, axis, angleOfArrow = cv2.fitEllipse(positionsOfArrow)
    # (x, y), (width, height), rect_angle = cv2.minAreaRect(positionsOfArrow)
    rectMin = cv2.minAreaRect(positionsOfArrow)
    rect = cv2.boxPoints(rectMin)

    # print rect
    # print rect[:,0]
    # print np.max(rect[:,0])
    if (np.min(rect[:,0]) < 0):
        return (0,0),-1
    if (np.max(rect[:,0]) > imageHeight):
        return (0,0),-1
    if (np.min(rect[:,1]) < 0):
        return (0,0),-1
    if (np.max(rect[:,1]) > imageWidth):
        return (0,0),-1
    
    if (
        # width*height > 360 or 
        rectMin[1][0]*rectMin[1][1] < 2):
        return (0,0),-1
    # print cv2.boxPoints(rect)
    # geometricCenter = (x + width/2,y + height/2)
    # geometricCenter = np.array(geometricCenter)
    geometricCenter = np.mean(rect, axis=0)
    if (geometricCenter[0] > 0.9 * (imageHeight)):
        return (0,0),-1
    # print geometricCenter


    # Barycenter
    barycenter = np.mean(positionsOfArrow, axis=1)
    # print barycenter, geometricCenter
    geometricCenterBarycenterVector = barycenter - geometricCenter
    # print geometricCenterBarycenterVector

    # verticalVector = np.array([0,-1])
    # angle = np.arctan2(np.linalg.norm(np.cross(geometricCenterBarycenterVector, verticalVector)), np.dot(geometricCenterBarycenterVector, verticalVector))
    # angle = (180*angle)/(np.pi)
    # print angle
    # print geometricCenter, angleOfArrow

    return geometricCenter, geometricCenterBarycenterVector

def decideEntrance(inputsOutputs):
    if len(inputsOutputs) <= 1:
        # print 'Solo una entrada'
        return [], []
        
    inOuts = np.array(inputsOutputs)
    # print 'inouts is', inOuts
    centerOfImage = np.array([imageHeight,int(imageWidth/2)])
    
    matrix = np.power(inOuts-centerOfImage,2)

    # print 'difference is', inOuts - centerOfImage

    # print 'matrix is', np.power(inOuts-centerOfImage,2)
    
    reduced = np.sum(matrix, axis = 1) # 1 para mantener el numero de salidas

    # print 'reduced is', reduced
    inPut = np.argmin(reduced)
    
    # print 'Voy a petar: ',inOuts

    inPuted = inOuts[inPut]
    
    inOuts = np.delete(inOuts,inPut,axis=0)

    # print 'Salidas= ',inOuts
    # print 'Entrada= ',inPuted

    return inPuted, inOuts


def decideExit(outputs, centerOfArrow, vectorOfArrow):
    # Consideramos outputs una lista de vectores en Numpy
    # np.arctan2(np.linalg.norm(np.cross(geometricCenterBarycenterVector, verticalVector)), np.dot(geometricCenterBarycenterVector, verticalVector))

    # print 'hereEEEEEEE'
    # print outputs
    # print centerOfArrow
    # print vectorOfArrow

    arrow = vectorOfArrow[0]


    outVectors = outputs - centerOfArrow
    angles = np.arccos(np.dot(outVectors, arrow) / (np.linalg.norm(outVectors) * np.linalg.norm(arrow)))
    # for v in outVectors:
    #     pEsc = np.arccos((outVectors[0]*vectorOfArrow[0] + outVectors[1]*vectorOfArrow[1]) / np.linalg.norm(outVectors) * np.linalg.norm(vectorOfArrow))
    # pass
    print 'angles', (angles * 180) / np.pi

    out = outputs[np.argmin(angles)]
    return out

times = []
while (capture.isOpened()):
    beg = time.time()

    ret, im = capture.read()
    # outIm.write(im)

    cv2.imshow('Real', im)

    # segmentation
    imHSV = im[((originalImageHeight - imageHeight)*shrinkFactor)::shrinkFactor, 0::shrinkFactor, :]
    print imHSV.shape
    imHSV = cv2.cvtColor(imHSV, cv2.COLOR_BGR2HSV)
    imHS = imHSV[:, :, (0, 1)]
    print imHS.shape

    def predictRow(i):
        # print segImg
        segImg[i] = clf.predict(imHS[i])

    [predictRow(i) for i in range(imHS.shape[0])]

    paletada = paleta[segImg]

    arrow = np.zeros(segImg.shape, dtype='uint8')
    line = np.zeros(segImg.shape, dtype='uint8')
    # 0 - line
    # 1 - floor
    # 2 - symbols
    # 3 - nothing - not used
    line[segImg == 0] = 1
    arrow[segImg == 2] = 1

    # arrow = cv2.erode(arrow, None, dst=arrow, iterations=2)
    # line = cv2.erode(line, None, dst=line, iterations=1)

    salidas = getSalidas(line)
    centerOfArrow, vectorOfArrow= getArrowPosition(arrow)


    # cv2.rectangle(paletada,(int(begRect[1]), int(begRect[0])),(int(endRect[1]), int(endRect[0])),(0,255,0),3)
    # paletada1, contours = cv2.findContours(arrow.astype('uint8')*255, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # paletada1 = cv2.drawContours(paletada, contours, -1, (0, 255, 0), 3)

    # if contours is not None or contours != [[[-1 -1 -1 -1]]]:
    #     print contours

    # print contours
    # if contours != None:
    #     for contour in contours:
    #         paletada[contour] = (255,255,255)

    if (vectorOfArrow is not -1):
        # print type(centerOfArrow), centerOfArrow
        # print type(vectorOfArrow), vectorOfArrow
        # cv2.ellipse(paletada,(int(centerOfArrow[1]), int(centerOfArrow[0])),(3,3),0,0,360,(255,0,255),-1)
        cv2.arrowedLine(
            paletada, 
            (int(centerOfArrow[1]), int(centerOfArrow[0])), 
            (int(vectorOfArrow[0][1]*20 + centerOfArrow[1]), int(vectorOfArrow[0][0]*20 + centerOfArrow[0])),
            # (0,0),
            (0,0,255),
            3)
        
        
    entrance, exits = decideEntrance(salidas)

    if (len(entrance) > 0):
        # cv2.circle(paletada, entrance, 5)
        cv2.ellipse(paletada,(int(entrance[1]), int(entrance[0])),(3,3),0,0,360,(0,255,255),-1)

        if (len(exits) > 0):
            for uniqueExit in exits:
                cv2.ellipse(paletada,(uniqueExit[1], uniqueExit[0]),(3,3),0,0,360,(255,0,255),-1)
            exitsArray = np.array(exits)
            if vectorOfArrow is not -1 and len(exits) > 1:
                selectedExit = decideExit(exitsArray, centerOfArrow, vectorOfArrow)
                cv2.ellipse(paletada,(selectedExit[1], selectedExit[0]),(3,3),0,0,360,(0,120,120),-1)
            else:
                selectedExit = exitsArray[0]
                cv2.ellipse(paletada,(selectedExit[1], selectedExit[0]),(3,3),0,0,360,(0,120,120),-1)


            # print 'entrance', entrance
            cv2.arrowedLine(
                paletada, 
                (int(entrance[1]), int(entrance[0])), 
                (int(selectedExit[1]), int(selectedExit[0])),
                # (0,0),
                (0,150,255),
                3)



    cv2.imshow("Segmentacion Euclid",
               cv2.cvtColor(paletada, cv2.COLOR_RGB2BGR))

    # cv2.imshow("Segmentacion Euclid",cv2.cvtColor(segImg,cv2.COLOR_RGB2BGR))

    # times.append(time.time() - beg)

    # print np.mean(np.array(times))

    cv2.waitKey(1)

    # outSeg.write(cv2.cvtColor(segImg,cv2.COLOR_RGB2BGR))