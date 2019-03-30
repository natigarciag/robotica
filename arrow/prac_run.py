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
segImg = np.empty((config.imageShape['height'] / shrinkFactor,
                   config.imageShape['width'] / shrinkFactor),
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
    geometricCenter, axis, angleOfArrow = cv2.fitEllipse(positionsOfArrow)

    # Barycenter
    barycenter = np.mean(positionsOfArrow, axis=1)
    geometricCenterBarycenterVector = barycenter - np.array(geometricCenter)

    verticalVector = np.array([0,-1])
    angle = np.arctan2(np.linalg.norm(np.cross(geometricCenterBarycenterVector, verticalVector)), np.dot(geometricCenterBarycenterVector, verticalVector))
    angle = (180*angle)/(np.pi)

    print angle

    



    # print geometricCenter, angleOfArrow

    return geometricCenter, angleOfArrow


times = []
while (capture.isOpened()):
    beg = time.time()

    ret, im = capture.read()
    # outIm.write(im)

    cv2.imshow('Real', im)

    # segmentation
    imHSV = im[0::shrinkFactor, 0::shrinkFactor, :]
    imHSV = cv2.cvtColor(imHSV, cv2.COLOR_BGR2HSV)
    imHS = imHSV[:, :, (0, 1)]

    def predictRow(i):
        segImg[i] = clf.predict(imHS[i])

    [predictRow(i) for i in range(imHS.shape[0])]

    arrow = np.zeros(segImg.shape, dtype='uint8')
    line = np.zeros(segImg.shape, dtype='uint8')
    # 0 - line
    # 1 - floor
    # 2 - symbols
    # 3 - nothing - not used
    line[segImg == 0] = 1
    arrow[segImg == 2] = 1

    arrow = cv2.erode(arrow, None, dst=arrow, iterations=1)
    # line = cv2.erode(line, None, dst=line, iterations=1)

    salidas = getSalidas(line)
    centerOfArrow, angleOfArrow = getArrowPosition(arrow)

    cv2.imshow("Segmentacion Euclid",
               cv2.cvtColor(paleta[segImg], cv2.COLOR_RGB2BGR))

    # cv2.imshow("Segmentacion Euclid",cv2.cvtColor(segImg,cv2.COLOR_RGB2BGR))

    # times.append(time.time() - beg)

    # print np.mean(np.array(times))

    cv2.waitKey(1)

    # outSeg.write(cv2.cvtColor(segImg,cv2.COLOR_RGB2BGR))