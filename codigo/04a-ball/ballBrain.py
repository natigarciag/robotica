from pyrobot.brain import Brain

import math

import cv2
from matplotlib import pyplot as plt
import numpy as np
import config
import datasetGenerator
import setup
#import imutils


# Autores:
# Luciano Garcia Giordano - 150245
# Gonzalo Florez Arias - 150048
# Salvador Gonzalez Gerpe - 150044

class BrainTestNavigator(
    Brain
    ):

    def setup(self):
        print 'setup'
        self.capture = setup.capture

        self.targetDistance = 40.0
        self.targetX = (setup.imageWidth // setup.shrinkFactor) / 2
        self.targetY = 120 // setup.shrinkFactor
        self.angularKp = 3.0
        self.angularKd = 3.0#5.0

        self.state = 'findBall'

        self.states = {
            'findBall': {
                'action': self.waitStopped
            },
            'followBall': {
                'action': self.followBall
            }
        }

        self.ballPosition = {}

    def waitStopped(self, hasBall, ballDistance, ballPosition):
        self.move(0.0, 0.0)

    def followBall(self, hasBall, ballDistance, ballPosition):
        # print 'distancia es',ballDistance
        angularDeviation = (ballPosition['x'] - self.targetX)/(setup.imageWidth // setup.shrinkFactor)

        linearDeviation = ballDistance - self.targetDistance
        # print angularDeviation

        turn = 0.0
        speed = 0.0
        if (hasBall):
            if math.fabs(angularDeviation) > 0.2:
                # turn = -self.Kp*angularDeviation
                diffAngularDev = math.fabs(angularDeviation) - math.fabs(self.prevAngularDeviation)
                if diffAngularDev == 0.0:
                    diffAngularDev = 1.1
                turn = (-self.angularKp*angularDeviation)/math.fabs(self.angularKd*diffAngularDev)
                # self.Kp += 0.1
                # print "Kp is:", self.Kp
                # print "turn is:", turn
                if (turn < -2.0): 
                    turn = -2.0
                elif (turn > 2.0):
                    turn = 2.0
            else:
                turn = 0.0

            if math.fabs(linearDeviation) > 5.0:
                diffLinearDeviation = math.fabs(linearDeviation) - math.fabs(self.prevLinearDeviation)
                if diffLinearDeviation == 0.0:
                    diffLinearDeviation = 5.0

                signalKeeper = 1 if linearDeviation > 0 else -1
                speed = signalKeeper * math.fabs(linearDeviation)/math.fabs(diffLinearDeviation)
                # print linearDeviation, diffLinearDeviation
                # speed = (self.linearKp*linearDeviation)
                # print speed
                if (speed < -1.0): 
                    speed = -1.0
                elif (speed > 1.0):
                    speed = 1.0
            else:
                speed = 0.0
            
        # Previous solutions:
        # if ballPosition['x'] > self.targetX:
        #     # Go right
        #     turn = -1.5
        # elif ballPosition['x'] < self.targetX:
        #     # go left
        #     turn = 1.5

        # speed = 0.0
        # turn = 0.0
        # if ballDistance < self.targetDistance:
        #     speed = -0.7
        # elif ballDistance > self.targetDistance:
        #     speed = 0.7


        self.prevAngularDeviation = angularDeviation

        self.move(speed, turn)

    def transitionState(self, hasBall, ballDistance, ballPosition):

        previousState = self.state
        
        if self.state is 'findBall':
            if hasBall:
                self.prevAngularDeviation = 0.0
                self.prevLinearDeviation = 0.0
                self.state = 'followBall'

        elif self.state is 'followBall':
            if not hasBall:
                self.state = 'findBall'


        if previousState != self.state:
            print "new state:", self.state
            # self.printSummary(previousState, hasBall, ballDistance, ballPosition)
            
    def printSummary(self, previousState, hasBall, ballDistance, ballPosition):
        print "Data for new state decision is", previousState, hasBall, ballDistance, ballPosition        
    
    def evaluateBall(self):
        #print 'Sigo vivo'
        ret, im = self.capture.read()

        # # Record video
        # outIm.write(im)
        if setup.drawAndRecordSchematicSegmentation:
            setup.rawVideoOutput.write(im)


	    # print ret

        # cv2.imshow('Real', im)

        # segmentation
        #print im

        # blurred = cv2.GaussianBlur(im, (7,7),0)
        # blurred = im
        imHSV = im[0::setup.shrinkFactor,0::setup.shrinkFactor,:]
        imHSV = cv2.cvtColor(imHSV, cv2.COLOR_BGR2HSV)
        imHS = imHSV[:,:,(0,1)]

        paleta = setup.paleta  

        def predictRow(i):
            reliabilityMapFullRow = setup.segmenter.clf.predict_proba(imHS[i])

            setup.segImg[i] = np.argmax(reliabilityMapFullRow, axis=1)
            
            reliabilityMaskSymbols = (reliabilityMapFullRow[:,1] > 0.7).astype('uint8')

            setup.segImg[i][setup.segImg[i] == 1] = reliabilityMaskSymbols[setup.segImg[i] == 1]

            # segImg[i] = paleta[self.clf.predict(imHS[i])]
            # setup.segImg[i] = setup.segmenter.predict(imHS[i])

        [predictRow(i) for i in range(imHS.shape[0])]
        
        showImage = np.zeros(setup.segImg.shape, dtype='uint8')
        showImage[setup.segImg==1] = 1

        # print showImage.shape
        segImg1 = cv2.erode(showImage, None, dst=showImage, iterations=1) # don't touch this
        # segImg1 = showImage
        if setup.drawAndRecordSchematicSegmentation:
            write = np.stack((segImg1,)*3, axis=-1)*255
            # print 'shape of schematic video',write.shape, segImg1.shape, setup.segImg.shape
            setup.schematicsVideoOutput.write(write)
        # print showImage.shape
        # print segImg1.shape
        # segImg1 = cv2.dilate(segImg1, None, iterations=2)
        # segImg1 = cv2.dilate(showImage, None, dst=showImage, iterations=2)

        # whereIsObject = np.all(segImg1 == 2, axis=-1)
        # print segImg1.shape
        # whereIsObject = np.all(segImg1 == 1, axis=-1)
        # print whereIsObject
        whereIsObjectPositions = np.where(segImg1==1)
        # print whereIsObjectPositions
        
        if whereIsObjectPositions[1].shape[0] != 0:
            minx = np.min(whereIsObjectPositions[1])
            maxx = np.max(whereIsObjectPositions[1])
            self.size = maxx - minx
            self.ballPosition['x'] = (maxx + minx) / 2.0
            if self.size == 0:
                self.size = 1.0
            hasBall = True

            dist0 = 20.0 # put fixed distance to the object here
            # diameter = 6.8 # put size of object here
            # paramF = 279.26
            # paramF = 64.0

            # dist = (paramF*diameter) / self.size

            size0 = 26*3

            dist = (size0*dist0)/self.size

            # print "minmax", minx, maxx, dist


            # print "paramF should be: ", (dist1*self.size)/diameter, "distance is: ", dist
        else:
            hasBall = False
            self.size = 30.0
            dist = -1.0

        

	    #print "imagen", segImg
        # cv2.imshow("Segmentacion Euclid",cv2.cvtColor(segImg,cv2.COLOR_RGB2BGR))
        # cv2.imshow("Segmentacion Euclid",showImage*255)
        
        
        # # Record video
        # outFrame = np.repeat((showImage*255)[:, :, np.newaxis], 3, axis=2)[:,:,:,0]
        # outSeg.write(outFrame)
	    
        
        
        # print outFrame.shape
        
        #print segImg[0][0][0]
    
        # cv2.waitKey(1)

        ballPosition = self.ballPosition


        return hasBall, dist, ballPosition

    def step(self):
        #print 'step'
        hasBall, ballDistance, ballPosition = self.evaluateBall()

        # Changes of state
        self.transitionState(hasBall, ballDistance, ballPosition)

        # Follow state
        self.states[self.state]['action'](hasBall, ballDistance, ballPosition)
    
def INIT(engine):
    print 'init1'
    assert (engine.robot.requires("range-sensor") and engine.robot.requires("continuous-movement"))
    print 'init2'
    # If we are allowed (for example you can't in a simulation), enable
    # the motors.
    # try:
    #     print 'init3'
    #     engine.robot.position[0]._dev.enable(1)
    #     print 'init4'
    # except AttributeError:
    #     print 'init5'
    #     pass
    #     print 'init6'

    print 'init7'

    return BrainTestNavigator('BrainTestNavigator', engine)




# brain = BrainTestNavigator()
# brain.setup()
# while(True):
#    brain.step()
