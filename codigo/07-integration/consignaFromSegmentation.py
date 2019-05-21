import cv2
import numpy as np
import math


def getSalidas(line, setup):
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

    # print 'Pixels', Result
    return Result

def getArrowPosition(arrow, setup):
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
    if (np.max(rect[:,0]) > setup.imageHeight):
        return (0,0),-1,[0,0],[[0,0]]
    if (np.min(rect[:,1]) < 0):
        return (0,0),-1,[0,0],[[0,0]]
    if (np.max(rect[:,1]) > setup.imageWidth):
        return (0,0),-1,[0,0],[[0,0]]
    
    if (rectMin[1][0]*rectMin[1][1] < 2):
        return (0,0),-1,[0,0],[[0,0]]
    geometricCenter = np.mean(rect, axis=0)
    if (geometricCenter[0] > 0.9 * (setup.imageHeight)):
        return (0,0),-1,[0,0],[[0,0]]

    # Barycenter
    barycenter = np.mean(positionsOfArrow, axis=1)
    geometricCenterBarycenterVector = barycenter - geometricCenter
    # print(geometricCenterBarycenterVector)

    # alternate = geometricCenterBarycenterVector[0][0]
    # geometricCenterBarycenterVector[0][0] = geometricCenterBarycenterVector[0][1]
    # geometricCenterBarycenterVector[0][1] = alternate

    return geometricCenter, geometricCenterBarycenterVector, geometricCenter, barycenter

def decideEntrance(inputsOutputs, setup):
    if len(inputsOutputs) <= 1:
        return [], []
        
    inOuts = np.array(inputsOutputs)
    centerOfImage = np.array([setup.imageHeight,setup.imageWidth//2])
    
    matrix = np.power(inOuts-centerOfImage,2)
    
    reduced = np.sum(matrix, axis = 1) # 1 para mantener el numero de salidas

    inPut = np.argmin(reduced)
    

    inPuted = inOuts[inPut]
    
    inOuts = np.delete(inOuts,inPut,axis=0)

    return inPuted, inOuts

def decideExit(outputs, centerOfArrow, vectorOfArrow, imageOnPaleta, setup):
    # Consideramos outputs una lista de vectores en Numpy

    arrow = vectorOfArrow[0]

    # print('outputs:', outputs)
    outVectors = outputs - centerOfArrow
    # print('vectorsToOutputs', outVectors)

    for vect in outVectors:
        cv2.arrowedLine(
            imageOnPaleta, 
            (int(centerOfArrow[1]), int(centerOfArrow[0])), 
            (int(centerOfArrow[1] + vect[1]), int(centerOfArrow[0] + vect[0])),
            (255,255,255),
            1)

    # print('arrow:', arrow)

    # print(outVectors)
    # print(np.linalg.norm(outVectors))
    # print(np.linalg.norm(arrow))
    # print('dot', np.dot(outVectors, arrow))
    # print('dot normalizado', np.dot(outVectors, arrow) / (np.linalg.norm(outVectors) * np.linalg.norm(arrow)))
    # angles = np.arccos(1 - np.abs(np.dot(outvectors, arrow) / (np.linalg.norm(outvectors) * np.linalg.norm(arrow))))
    def calculateAngle(outVector, arrow):
        # angle = np.arccos(np.dot(outVector, arrow) / (np.linalg.norm(outVector) * np.linalg.norm(arrow)))
        # return angle
        return np.arccos(np.dot(outVector, arrow) / (np.linalg.norm(outVector) * np.linalg.norm(arrow)))
        
    angles = np.array([calculateAngle(outVectors[i], arrow) for i in range(outVectors.shape[0])])

    # print('angles:', (angles * 180) / math.pi)

    out = outputs[np.argmin(angles)]
    return out

def chooseIndexOfMostCentralExit(exitsArray, setup):
    # exitsArray is an np array of exits. Choose the most "central top" one

    centerOfImage = np.array([0,setup.imageWidth//2])
    
    matrix = np.power(exitsArray-centerOfImage,2)
    reduced = np.sum(matrix, axis = 1)
    indexOfOutput = np.argmin(reduced)
    
    return indexOfOutput

def calculateConsignaFromVector(vector, distanceToEntrance, previousData, setup, entrance):
    # Considering vector in the YX inverted coordinates (as in the matrices themselves)
    consignaArray = vector
    # print(vector)
    verticalVector = np.array([-1, 0])
    consignaAngle = np.arccos(np.dot(consignaArray, verticalVector) / (np.linalg.norm(consignaArray) * np.linalg.norm(verticalVector)))
    if (consignaArray[1] < 0):
        consignaAngle = -consignaAngle
    consignaAngle = (consignaAngle * 180) / np.pi
    if (consignaArray[0] > 0):
        consignaAngle = 180 - consignaAngle

    # print(consignaAngle)

    # Define Kd depending on the necessity
    if ((distanceToEntrance > 0 and consignaAngle > 0) or (distanceToEntrance < 0 and consignaAngle < 0)):
        # rotate a lot more than normal
        rotationKd = 0.1
    elif ((distanceToEntrance > 0 and consignaAngle < 0) or (distanceToEntrance < 0 and consignaAngle > 0)):
        # go as normal
        rotationKd = 0.2
    else:
        rotationKd = 0.25

    # if consignaAngle > 0:
    #     # go right


    # else:
    #     # go left

    # Calculating the consigna
    rotationD = previousData['angle'] - consignaAngle
    signalKeeper = ((consignaAngle // consignaAngle) if consignaAngle != 0 else 1)
    # print('angle and distance', consignaAngle, distanceToEntrance)
    rotation = signalKeeper*math.fabs((consignaAngle*0.003 + distanceToEntrance*0.001)/(math.fabs(rotationD)*rotationKd))
    # print rotation

    if (rotation > 1):
        rotation = 1
    elif (rotation < -1):
        rotation = -1

    speed = -0.5*math.fabs(rotation) + 1

    if (speed > 1):
        speed = 1
    elif (speed < 0):
        speed = 0
    
    rotationRobot = -rotation

    # print round(speed, 1), round(rotationRobot,1)

    previousData['angle'] = consignaAngle
    previousData['distance'] = distanceToEntrance



    rotation = consignaAngle/10

    # if consignaAngle > 20 and distanceToEntrance > 0:
    #     rotation -= 0.2
    # elif consignaAngle < -20 and distanceToEntrance < 0:
    #     rotation += 0.2


    if (rotation > 0):
        rotation = 0.4
    elif (rotation < 0):
        rotation = -0.4

    rotation = -rotation

    speed = -0.5*math.fabs(rotation) + 1

    if (speed > 1):
        speed = 1
    elif (speed < 0):
        speed = 0


    return speed, rotation



# parameter calculation
da = 0.2
d1 = 0.45
phi = 0.5

b = (phi - (math.pow(da,2)/math.pow(d1,2)))/(da - (math.pow(da,2)/d1))
a = (1 - b*d1)/math.pow(d1,2)

Kd = 25

def calculateConsignaFromExitDistance(distanceToExitPercentage, angle, previousData, setup):
    signalKeeper = (1.0 if distanceToExitPercentage > 0 else (-1.0 if distanceToExitPercentage < 0 else 0.0))
    distanceToExit = (distanceToExitPercentage if distanceToExitPercentage > 0 else -distanceToExitPercentage)

    turn = 0
    if distanceToExit > d1:
        turn = signalKeeper*1
    else:
        turn = signalKeeper * (math.pow(distanceToExit,2)*a + distanceToExit*b)

    derivativeTerm = previousData['distance'] - distanceToExitPercentage
    # print 'derivative term', derivativeTerm
    if derivativeTerm > 0.3:
        turn = turn / (math.fabs(derivativeTerm) * Kd)

    previousData['distance'] = distanceToExitPercentage

    speed = calculateForwardSpeedFromTurn(turn)

    if speed > 1:
        speed = 1
    elif speed < 0:
        speed = 0

    
    # print a,b,turn,distanceToExit,speed

    turn = -turn
    # print speed, turn

    return speed, turn


def calculateForwardSpeedFromTurn(turn):
    return -0.9*math.fabs(turn) + 1

def calculateConsignaFullProcess(line, arrow, imageOnPaleta, previousData, setup, entrance, exits):
    centerOfArrow, vectorOfArrow, geometricCenterArrow, barycenterArrow = getArrowPosition(arrow, setup)

    if (len(exits) > 0):
        exitsArray = np.array(exits)
        if vectorOfArrow is not -1 and len(exits) > 1:
            # When there's an arrow and many exits, decide from the arrow
            selectedExit = decideExit(exitsArray, centerOfArrow, vectorOfArrow, imageOnPaleta, setup)
        else:
            # When there are many exits and no arrows, choose the most central one
            selectedExit = exits[chooseIndexOfMostCentralExit(exitsArray, setup)]
        
        
        if entrance[0] < selectedExit[0]:
            # vector is most likely inverted
            # print("inverted????", entrance, selectedExit)
            vector = np.array([entrance[0] - selectedExit[0], entrance[1] - selectedExit[1]])
        else:
            vector = np.array([selectedExit[0] - entrance[0], selectedExit[1] - entrance[1]])

        distanceToEntrance = entrance[1] - setup.imageWidth//2

        distanceToExitPercentage = (selectedExit[1] + 0.0 - setup.imageWidth/2)/setup.imageWidth


        consignaArray = vector
        # print(vector)
        verticalVector = np.array([-1, 0])
        angle = np.arccos(np.dot(vector, verticalVector) / (np.linalg.norm(vector) * np.linalg.norm(verticalVector)))
        if (vector[1] < 0):
            angle = -angle
        angle = (angle * 180) / np.pi
        if (vector[0] > 0):
            angle = 180 - angle

        # print 'angle is:', angle

        speed, rotation = calculateConsignaFromExitDistance(distanceToExitPercentage, angle, previousData, setup)
    else:
        speed, rotation = None, None

    if setup.showSegmentedImage or setup.drawAndRecordSchematicSegmentation:
        if (vectorOfArrow is not -1):
            cv2.arrowedLine(
                imageOnPaleta, 
                (int(centerOfArrow[1]), int(centerOfArrow[0])), 
                (int(vectorOfArrow[0][1]*20 + centerOfArrow[1]), int(vectorOfArrow[0][0]*20 + centerOfArrow[0])),
                (0,0,255),
                1)
            # print(geometricCenterArrow)
            # print(barycenterArrow)
            imageOnPaleta[int(geometricCenterArrow[0]), int(geometricCenterArrow[1]),0] = 255
            imageOnPaleta[int(geometricCenterArrow[0]), int(geometricCenterArrow[1]),1] = 255
            imageOnPaleta[int(geometricCenterArrow[0]), int(geometricCenterArrow[1]),2] = 255
            imageOnPaleta[int(barycenterArrow[0][0]), int(barycenterArrow[0][1]),0] = 255
            imageOnPaleta[int(barycenterArrow[0][0]), int(barycenterArrow[0][1]),1] = 255
            imageOnPaleta[int(barycenterArrow[0][0]), int(barycenterArrow[0][1]),2] = 255

        if (len(entrance) > 0):
            # Since there's an entrance, show it on the image
            cv2.ellipse(imageOnPaleta,(int(entrance[1]), int(entrance[0])),(3,3),0,0,360,(0,255,255),-1)

            if (len(exits) > 0):
                for uniqueExit in exits:
                    cv2.ellipse(imageOnPaleta,(uniqueExit[1], uniqueExit[0]),(3,3),0,0,360,(255,0,255),-1)
                cv2.ellipse(imageOnPaleta,(selectedExit[1], selectedExit[0]),(3,3),0,0,360,(0,120,120),-1)
                if entrance[0] < selectedExit[0]:
                    cv2.arrowedLine(
                        imageOnPaleta, 
                        (int(selectedExit[1]), int(selectedExit[0])),
                        (int(entrance[1]), int(entrance[0])), 
                        (255,255,0),
                        1)
                else:
                    cv2.arrowedLine(
                        imageOnPaleta, 
                        (int(entrance[1]), int(entrance[0])), 
                        (int(selectedExit[1]), int(selectedExit[0])),
                        (255,255,0),
                        1)
        
        if speed != None:
            text = 'speed: ' + str(round(speed,4))
            cv2.putText(imageOnPaleta,text,(10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255),1,cv2.LINE_AA)
            text = 'turn: ' + str(round(rotation,4))
            cv2.putText(imageOnPaleta,text,(10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255),1,cv2.LINE_AA)
            text = 'distance: ' + str(round(distanceToExitPercentage,4))
            cv2.putText(imageOnPaleta,text,(10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255),1,cv2.LINE_AA)
            

    return speed, rotation, len(exits)

def predictShapeIfShape(arrow, setup):
    touchesEdges = setup.touchingEdges(arrow, 1)

    if(np.sum(arrow) > int(200 / (setup.shrinkFactor*setup.shrinkFactor))):
        if (not touchesEdges):
            moments = cv2.HuMoments(cv2.moments(arrow)).flatten()
            # print(moments)
            predictedShape = setup.symbolClassifier.predict(np.array([moments]))
            # print(predictedShape, namesOfTheShapes[int(predictedShape[0])])
            # print(predictedShape)
            return setup.namesOfTheShapes[int(predictedShape[0])]
        else:
            return 'touches edges'
    else:
        return None


def calculateEntranceAndExits(line, previousData, setup):
    salidas = getSalidas(line, setup)
    entrance, exits = decideEntrance(salidas, setup)

    return entrance, exits