# from pyrobot.brain import Brain

import math

import cv2
from matplotlib import pyplot as plt
import numpy as np
import config
import datasetGenerator
import clasificadorEuc
import imutils

# Autores:
# Luciano Garcia Giordano - 150245
# Gonzalo Florez Arias - 150048
# Salvador Gonzalez Gerpe - 150044




class BrainTestNavigator(
    Brain
    ):

    NO_FORWARD = 0
    SLOW_FORWARD = 0.1
    MED_FORWARD = 0.5
    FULL_FORWARD = 1.0

    NO_TURN = 0
    MED_LEFT = 0.5
    HARD_LEFT = 1.0
    MED_RIGHT = -0.5
    HARD_RIGHT = -1.0
    
    NO_ERROR = 0
    
    def setup(self):
        print 'setup'
        self.capture = cv2.VideoCapture('./videos/video.mp4')
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        self.clf = clasificadorEuc.Clasificador(datasetGenerator.shapeD)
        self.clf.train()

        self.targetDistance = 30.0
        self.targetX = 160
        self.targetY = 120

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
        turn = 0.0
        if ballPosition['x'] > targetX:
            # Go right
            turn = -0.5
        elif ballPosition['x'] < targetX:
            # go left
            turn = 0.5


        speed = 0.0
        if ballDistance < targetDistance:
            speed = -0.5
        elif ballDistance > targetDistance:
            speed = 0.5

        self.move(speed, turn)

    def followLine(self, hasBall, ballDistance, ballPosition):
        if (hasLine):
            Kp = 0.6*math.fabs(lineDistance)
            Kd = (math.fabs(lineDistance) - math.fabs(self.prevDistance))
            # print "Kd sale: ",Kd

            # self.Ki = self.Ki + lineDistance
            if Kd == 0:
                turnSpeed = self.previousTurn
            else:
                turnSpeed =  5*((Kp * lineDistance)/((searchRange - math.fabs(Kd)) * searchRange))
            
            turnSpeed = min(turnSpeed, 1)

            # The sharper the turn, the slower the robot advances forward
            # forwardVelocity = min(((searchRange - math.fabs(Kd)) * searchRange) / 180, 1)
            parNormalizacion = 100
            parA = 3.5
            parB = -2.8
            # Kd = Kd*5
            forwardVelocity = max(min(parA*((searchRange - 0.99*math.fabs(Kd)) * searchRange)**2 / (parNormalizacion**2) + parB*((searchRange - 0.99*math.fabs(Kd)) * searchRange) / (parNormalizacion), 1),0)

            maxSpeed = 1.0 if (front > 1.0 and left > 0.5 and right > 0.5) else 0.2
            if forwardVelocity > maxSpeed:
                forwardVelocity = maxSpeed

            self.previousTurn = turnSpeed

            # print "vel:",forwardVelocity,"turn:",turnSpeed, "Kd:",Kd
            self.move(forwardVelocity,turnSpeed)

        # elif self.firstStep:
        #     self.firstStep = False
        else:

            # if we can't find the line we just go back, this isn't very smart (but definitely better than just stopping
            turnSpeed = 0.8 if self.previousTurn > 0 else -0.8
            if self.lastTurn != 0:
                self.move(-0.2, -(self.lastTurn))
                self.lastTurn = 0
            else:
                self.move(-0.2,turnSpeed)
                self.lastTurn = turnSpeed

        self.prevDistance = lineDistance

    def transitionState(self, hasBall, ballDistance, ballPosition):

        previousState = self.state
        
        if self.state is 'findBall':
            if hasBall:
                self.state = 'followBall'

        elif self.state is 'followBall':
            if not hasBall:
                self.state = 'findBall'


        if previousState != self.state:
            print "new state:", self.state
        self.printSummary(previousState, hasBall, ballDistance, ballPosition)
            
    def printSummary(self, previousState, hasBall, ballDistance, ballPosition):
        print "Data for new state decision is", previousState, hasBall, ballDistance, ballPosition        
    
    def evaluateBall(self):
        print 'Sigo vivo'
        ret, im = self.capture.read()

        # cv2.imshow('Real', im)

        # segmentation
        print im

        blurred = cv2.GaussianBlur(im, (7,7),0)
        imHSV = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        imHS = imHSV[:,:,(0,1)]

        paleta = np.array([[0,0,0],[0,0,255],[0,255,0],[255,0,0]],dtype='uint8')  

        segImg = np.array([paleta[self.clf.predict(imHS[i])] for i in range(imHS.shape[0])], dtype='uint8')

        whereIsObject = np.all(segImg == [0,255,0], axis=-1)
        whereIsObjectPositions = np.where(whereIsObject==True)
        
        if whereIsObjectPositions[1].shape[0] != 0:
            minx = np.min(whereIsObjectPositions[1])
            maxx = np.max(whereIsObjectPositions[1])
            self.size = maxx - minx
            self.ballPosition['x'] = (maxx + minx) / 2.0
            if self.size == 0:
                self.size = 1.0
            hasBall = True
        else:
            hasBall = False
            self.size = 30.0

        # dist = 37.0 # put fixed distance to the object here
        diameter = 12.0 # put size of object here
        # print dist*size/diameter #uncomment to see your value of paramF
        paramF = 292.9

        dist = paramF*diameter / self.size
        # print dist


        # cv2.imshow("Segmentacion Euclid",cv2.cvtColor(segImg,cv2.COLOR_RGB2BGR))
    
        # cv2.waitKey(1)

        ballPosition = self.ballPosition


        return hasBall, dist, ballPosition

    def step(self):
        print 'step'
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
#     brain.step()