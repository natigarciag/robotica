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
        self.firstStep = True
        pass

    def step(self):
        hasLine,lineDistance,searchRange = eval(self.robot.simulation[0].eval("self.getLineProperties()"))
        print "I got from the simulation",hasLine,lineDistance,searchRange

        # if (hasLine or not self.firstStep):
        if (hasLine):
            Kp = 0.6*math.fabs(lineDistance)
            Kd = (math.fabs(lineDistance) - math.fabs(self.prevDistance))
            print "Kd sale: ",Kd

            # self.Ki = self.Ki + lineDistance

            turnSpeed =  5*((Kp * lineDistance)/((searchRange - math.fabs(Kd)) * searchRange))
            turnSpeed = min(turnSpeed, 1)

            # The sharper the turn, the slower the robot advances forward
            # forwardVelocity = min(((searchRange - math.fabs(Kd)) * searchRange) / 180, 1)
            parA = 150
            forwardVelocity = min(0.6*((searchRange - math.fabs(Kd)) * searchRange)**2 / (parA**2) + 0.5*((searchRange - math.fabs(Kd)) * searchRange) / (parA), 1)

            self.previousTurn = turnSpeed

            self.move(forwardVelocity,turnSpeed)

        # elif self.firstStep:
        #     self.firstStep = False
        else:

            # if we can't find the line we just go back, this isn't very smart (but definitely better than just stopping
            turnSpeed = 0.1 if self.previousTurn > 0 else -0.1
            self.move(-0.1, turnSpeed)

        self.prevDistance = lineDistance
        

        # if (hasLine):
        # if (lineDistance > self.NO_ERROR):
        # 	self.move(self.FULL_FORWARD,self.HARD_LEFT)
        # elif (lineDistance < self.NO_ERROR):
        # 	self.move(self.FULL_FORWARD,self.HARD_RIGHT)
        # else:
        # 	self.move(self.FULL_FORWARD,self.NO_TURN)
        # else:
        # # if we can't find the line we just stop, this isn't very smart
        # self.move(self.NO_FORWARD,self.NO_TURN)
        # 
    
def INIT(engine):
    assert (engine.robot.requires("range-sensor") and engine.robot.requires("continuous-movement"))

    # If we are allowed (for example you can't in a simulation), enable
    # the motors.
    try:
        engine.robot.position[0]._dev.enable(1)
    except AttributeError:
        pass
        
    return BrainTestNavigator('BrainTestNavigator', engine)
