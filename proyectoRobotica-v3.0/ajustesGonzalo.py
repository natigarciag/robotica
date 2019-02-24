from pyrobot.brain import Brain

import math

class BrainTestNavigator(Brain):

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
        self.prevDistance = 10.0
        self.Ki = 0.0   
        self.previousTurn = 0
        self.state = 'searchLine'
        self.searchLine = True
        self.prevDistanceToObstacle = 1.0

        self.states = {
            'searchLine': {
                'action': self.followLine
            },
            'approachLine': {
                'action': self.followLine
            },
            'followLine': {
                'action': self.followLine
            },
            'circleObjectOnRight': {
                'action': self.followObjectOnRight
            },
            'leaveObjectOnRight': {
                'action': self.leaveObjectOnRight
            }
        }

        pass

    def followLine(self, hasLine,lineDistance,searchRange,hard_left, left, front, right, hard_right):
        # if (hasLine or not self.firstStep):
        if (hasLine):
            Kp = 0.6*math.fabs(lineDistance)
            Kd = (math.fabs(lineDistance) - math.fabs(self.prevDistance))
            # print "Kd sale: ",Kd

            # self.Ki = self.Ki + lineDistance

            turnSpeed =  5*((Kp * lineDistance)/((searchRange - math.fabs(Kd)) * searchRange))
            turnSpeed = min(turnSpeed, 1)

            # The sharper the turn, the slower the robot advances forward
            # forwardVelocity = min(((searchRange - math.fabs(Kd)) * searchRange) / 180, 1)
            parNormalizacion = 100
            parA = 3.5
            parB = -2.8
            # Kd = Kd*5
            forwardVelocity = max(min(parA*((searchRange - 0.99*math.fabs(Kd)) * searchRange)**2 / (parNormalizacion**2) + parB*((searchRange - 0.99*math.fabs(Kd)) * searchRange) / (parNormalizacion), 1),0)

            self.previousTurn = turnSpeed

            # print "vel:",forwardVelocity,"turn:",turnSpeed, "Kd:",Kd
            self.move(forwardVelocity,turnSpeed)

        # elif self.firstStep:
        #     self.firstStep = False
        else:

            # if we can't find the line we just go back, this isn't very smart (but definitely better than just stopping
            turnSpeed = 0.1 if self.previousTurn > 0 else -0.1
            self.move(-0.1, turnSpeed)

        self.prevDistance = lineDistance

    def followObjectOnLeft(self, hasLine,lineDistance,searchRange,hard_left, left, front, right, hard_right):
        self.move(0,0.3)

    def followObjectOnRight(self,hasLine,lineDistance,searchRange,hard_left, left, front, right, hard_right):
        lineDistance = min([front, left, right]) - 0.6
        # print "Distance to object is:",lineDistance
        Kp = 0.6*math.fabs(lineDistance)
        Kd = lineDistance - self.prevDistanceToObstacle
        turnSpeed =  -3*((Kp * lineDistance)/math.fabs(Kd))
        turnSpeed = max(min(turnSpeed, 1),-1)

        parA = 0.6 #3.8
        parB = 1.3 #-2.8
        Kd = Kd*10
        maxSpeed = 0.5
        x = max(0,min(1,math.fabs(Kd)))
        rawForwardVelocity = parA*(x)**2 + parB*(x)
        forwardVelocity = max(min(rawForwardVelocity, maxSpeed),0)

        self.previousTurn = turnSpeed
        self.prevDistanceToObstacle = lineDistance

        # print "vel:",rawForwardVelocity,"turn:",turnSpeed, "Kd:",Kd
        self.move(forwardVelocity,turnSpeed)

    def leaveObjectOnRight(self,hasLine,lineDistance,searchRange,hard_left, left, front, right, hard_right):
        self.move(0.2, 0.8)

    def transitionState(self, hasLine,lineDistance,searchRange,hard_left, left, front, right, hard_right):
        previousState = self.state
        absDist = math.fabs(lineDistance)

        if self.state is 'searchLine':
            if hasLine:
                self.state = 'followLine'

        elif self.state is 'approachLine':
            if (hasLine and absDist < 0.3):
                self.state = 'followLine'
            
            if (front < 0.7 or left < 0.1 or right < 0.1):
                self.state = 'circleObjectOnRight'
                self.notFoundLineFor = 10

        elif self.state is 'followLine':
            if (front < 0.7 or left < 0.1 or right < 0.1):
                self.state = 'circleObjectOnRight'
                self.notFoundLineFor = 10

        elif self.state is 'circleObjectOnRight':
            if hasLine and self.searchLine is True and self.notFoundLineFor < 0:
                self.state = 'leaveObjectOnRight'
                self.searchLine = False
                self.prevDistance = lineDistance
            elif not hasLine:
                self.searchLine = True
                self.notFoundLineFor -= 1
                # print 'not found for',self.notFoundLineFor
            

        elif self.state is 'leaveObjectOnRight':
            if hasLine and absDist < 0.95:
                self.state = 'approachLine'
            self.printSummary(previousState, hasLine,lineDistance,searchRange,hard_left, left, front, right, hard_right)
         

        if previousState != self.state:
            print "new state:", self.state
            self.printSummary(previousState, hasLine,lineDistance,searchRange,hard_left, left, front, right, hard_right)
            
    def printSummary(self, previousState, hasLine,lineDistance,searchRange,hard_left, left, front, right, hard_right):
        print "Data for new state decision is", previousState, hasLine,lineDistance,searchRange,hard_left, left, front, right, hard_right        
    

    def step(self):
        hasLine,lineDistance,searchRange = eval(self.robot.simulation[0].eval("self.getLineProperties()"))
        
        front = min([s.distance() for s in self.robot.range["front"]])
        left = min([s.distance() for s in self.robot.range["left-front"]])
        right = min([s.distance() for s in self.robot.range["right-front"]])
        hard_left = self.robot.range[0].distance()
        hard_right = self.robot.range[7].distance()

        # Changes of state
        self.transitionState(hasLine,lineDistance,searchRange,hard_left, left, front, right, hard_right)

        # Follow state
        self.states[self.state]['action'](hasLine,lineDistance,searchRange,hard_left, left, front, right, hard_right)
    
def INIT(engine):
    assert (engine.robot.requires("range-sensor") and engine.robot.requires("continuous-movement"))

    # If we are allowed (for example you can't in a simulation), enable
    # the motors.
    try:
        engine.robot.position[0]._dev.enable(1)
    except AttributeError:
        pass
        
    return BrainTestNavigator('BrainTestNavigator', engine)
