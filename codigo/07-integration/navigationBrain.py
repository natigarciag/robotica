from pyrobot.brain import Brain

import math

import cv2
from matplotlib import pyplot as plt
import numpy as np
import config
import setup
#import imutils
import segmentLinesAndSymbolsFromImage
import consignaFromSegmentation
import datetime


# Autores:
# Luciano Garcia Giordano - 150245
# Gonzalo Florez Arias - 150048
# Salvador Gonzalez Gerpe - 150044

# Record video
# outIm = cv2.VideoWriter('./videos/demoBolaImagen.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, (320, 240))
# outSeg = cv2.VideoWriter('./videos/demoBolaSegm.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, (320/4, 240/4))


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

    # def move(self,a,b):
	#     pass
    
    def setup(self):
        print 'setup'

        self.targetDistance = 40.0
        self.targetX = setup.imageWidth // 2
        self.targetY = 120
        self.angularKp = 3.0 # turns more the higher
        self.angularKd = 15.0 # turns more slowly the lower

        self.linearKp = 0.08
        self.linearKd = 0.05
        self.followArrowProcedureFinished = False
        self.followArrowProcedureCounter = 0
        self.followArrowObject = {
            'fixed': False,
            'mostRotation': 0
        }

        self.state = 'followLine'

        self.states = {
            'findLine': {
                'action': self.findLine
            },
            'followLine': {
                'action': self.followLine
            },
            'followArrow': {
                'action': self.followArrow
            }
        }

        self.ballPosition = {}

        self.previousData = {
            "angle": 0,
            "distance": 0,
            "turn": 0,
            "searchLineTurnDirectionCounter": 0
        }

    def findLine(self, arrow, line, entrance, exits, imageOnPaleta):
        turnSpeed = 0.8 if self.previousData['turn'] > 0 else -0.8
        if (self.previousData['searchLineTurnDirectionCounter'] % 10) > 5:
            turnSpeed = 0.8 if self.previousData['turn'] < 0 else -0.8

        
        if self.previousData['turn'] > 0:
            self.move(-0.2, -turnSpeed)
        else:
            self.move(-0.2, turnSpeed)

        self.previousData['searchLineTurnDirectionCounter'] += 1

    def waitStopped(self, arrow, line, entrance, exits, imageOnPaleta):
        self.move(0.0, 0.0)

    def followArrow(self, arrow, line, entrance, exits, imageOnPaleta):
        speed, rotation, numberOfExits = consignaFromSegmentation.calculateConsignaFullProcess(line, arrow, imageOnPaleta, self.previousData, setup, entrance, exits)
        if rotation != None and speed != None and np.abs(rotation) > np.abs(self.followArrowObject['mostRotation']):
            # print 'changing rotation to', rotation
            self.followArrowObject['mostRotation'] = rotation
        
        positionsOfRed = np.where(arrow == 1)
        centralRedPosition = np.mean(positionsOfRed, axis=1)
        
        if (setup.touchingEdges(arrow, 10) or centralRedPosition[0] < 0.5 * setup.imageHeight) and self.followArrowObject['fixed'] is False:
            centralHorizontalPoint = (centralRedPosition[1] - (setup.imageWidth/2))/setup.imageWidth
            
            signalKeeper = (1.0 if centralHorizontalPoint > 0 else (-1.0 if centralHorizontalPoint < 0 else 0.0))
            turnPar = (centralHorizontalPoint if centralHorizontalPoint > 0 else -centralHorizontalPoint)

            turn = -1 * signalKeeper * (math.pow(turnPar,2)*consignaFromSegmentation.a + turnPar*consignaFromSegmentation.b)
            self.move(-math.fabs(turn) + 1 , turn)
            if np.sum(arrow) < 50 or len(exits) > 1:
                self.followArrowProcedureFinished = True
        else:
            if (np.sum(arrow) > 50) and self.followArrowObject['fixed'] is False:
                date = datetime.datetime.now()
                cv2.imwrite('imagenConsignaFlecha_' + date.strftime("%m_%d_%H_%M") + '.png', imageOnPaleta)
                print 'consigna fixed at:', speed, self.followArrowObject['mostRotation']
                self.followArrowObject['fixed'] = True
                self.followArrowObject['speed'] = consignaFromSegmentation.calculateForwardSpeedFromTurn(self.followArrowObject['mostRotation'])
                self.followArrowObject['rotation'] = self.followArrowObject['mostRotation']
            
            if (self.followArrowObject['fixed'] is True):
                self.move(self.followArrowObject['speed'], self.followArrowObject['rotation'])
                if (len(exits) < 2):
                    self.followArrowProcedureFinished = True
                    self.followArrowObject['fixed'] = False
                    self.followArrowObject['mostRotation'] = 0
        
        self.previousData['turn'] = rotation

    def followLine(self, arrow, line, entrance, exits, imageOnPaleta):
        speed, rotation, numberOfExits = consignaFromSegmentation.calculateConsignaFullProcess(line, arrow, imageOnPaleta, self.previousData, setup, entrance, exits)
        # print speed, rotation
        if speed != None and rotation != None:
            # print 'sending command', speed, rotation
            self.move(speed, rotation)
        else:
            self.move(0,0)

        self.previousData['turn'] = rotation
        
    def transitionState(self, arrow, line, entrance, exits):
        previousState = self.state

        if previousState != 'findLine':
            self.previousData['searchLineTurnDirectionCounter'] = 0
        
        if previousState != 'followArrow':
            self.followArrowProcedureFinished = False
            self.followArrowProcedureCounter = 0
        
        if self.state is 'followLine':
            if len(exits) == 0 and len(entrance) == 0:
                self.state = 'findLine'
            elif len(exits) > 1:
                if np.sum(arrow) > 200:
                    self.state = 'followArrow'

        elif self.state is 'findLine':
            if len(entrance) > 0:
                self.state = 'followLine'

        elif self.state is 'followArrow':
            if len(entrance) == 1 and self.followArrowProcedureFinished == True:
                self.state = 'followLine'
            elif len(entrance) == 0 and self.followArrowProcedureFinished == True:
                self.state = 'findLine'

        if previousState != self.state:
            print "new state:", self.state
            
    def printSummary(self, previousState, hasBall, ballDistance, ballPosition):
        print "Data for new state decision is", previousState, hasBall, ballDistance, ballPosition        
    
    def evaluateSituation(self):
        arrow, line, imageOnPaleta = segmentLinesAndSymbolsFromImage.fetchImageAndSegment(setup)

        entrance, exits = consignaFromSegmentation.calculateEntranceAndExits(line, self.previousData, setup)

        if (len(exits) < 2):
            shapeName = consignaFromSegmentation.predictShapeIfShape(arrow, setup)
            if (not (shapeName == None or shapeName == 'touches edges' or shapeName == 'nothing')) and numberOfExits <= 1:
                print(shapeName)
                cv2.putText(imageOnPaleta,shapeName,(10,160//setup.segmentedImageShrinkFactor), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255),1,cv2.LINE_AA)

        return arrow, line, entrance, exits, imageOnPaleta

    def step(self):
        arrow, line, entrance, exits, imageOnPaleta = self.evaluateSituation()

        # Changes of state
        self.transitionState(arrow, line, entrance, exits)

        # Follow state
        self.states[self.state]['action'](arrow, line, entrance, exits, imageOnPaleta)

        if setup.drawAndRecordSchematicSegmentation:
            setup.schematicsVideoOutput.write(cv2.cvtColor(imageOnPaleta, cv2.COLOR_RGB2BGR))
        
        
    
def INIT(engine):
    assert (engine.robot.requires("range-sensor") and engine.robot.requires("continuous-movement"))
    return BrainTestNavigator('BrainTestNavigator', engine)




# brain = BrainTestNavigator()
# brain.setup()
# while(True):
#    brain.step()
