from pyrobot.brain import Brain

import math

class BrainTestNavigator(Brain):


  def setup(self):
    self.prevDistance = 0.0
    self.Ki = 0.0
    pass

  def step(self):
    hasLine,lineDistance,searchRange = eval(self.robot.simulation[0].eval("self.getLineProperties()"))
    print "I got from the simulation",hasLine,lineDistance,searchRange

    if (hasLine):
	Kp = lineDistance
	Kd = lineDistance - self.prevDistance
	self.Ki = self.Ki + lineDistance	

	Vt = lineDistance/searchRange 
	Vf = max(0,1-math.fabs(Vt*1.5))
	self.move(Vf,Vt)
    else:
      # if we can't find the line we just go back, this isn't very smart (but definitely better than just stopping
      self.move(-0.4,0)

    self.prevDistance = lineDistance
 
def INIT(engine):
  assert (engine.robot.requires("range-sensor") and
	  engine.robot.requires("continuous-movement"))

  # If we are allowed (for example you can't in a simulation), enable
  # the motors.
  try:
    engine.robot.position[0]._dev.enable(1)
  except AttributeError:
    pass

  return BrainTestNavigator('BrainTestNavigator', engine)
