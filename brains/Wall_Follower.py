# robot goes forward and then slows and continues following the wall
import sys
from pyrobot.brain import Brain  
   
# importado por mi cuenta

import numpy as np

class Wall_Follower(Brain): 
	# Give the front two sensors, decide the next move sys.maxsize

	def determineMove(self, front, left, right):
		if front < 0.5:
			return(0, .8)
		#elif self.robot.range[3].distance() > 0.5 and self.robot.range[4].distance() > 0.5:
		#	return(1,0)
		elif self.robot.range[7].distance() > 0.2 and self.robot.range[7].distance() < 0.25 and self.robot.range[6].distance() > 0.27 and self.robot.range[6].distance() < 0.32:
			return(1, 0)
		elif self.robot.range[7].distance() > 0.25 or self.robot.range[6].distance() > 0.32:
			return(0.2,-.4)
		elif self.robot.range[7].distance() < 0.2 or self.robot.range[6].distance() < 0.27:
			return(0.2,.4)
		else:  
			return(0.8, 0)

	def step(self):  
		front = min([s.distance() for s in self.robot.range["front"]])
		left = min([s.distance() for s in self.robot.range["left-front"]])
		right = min([s.distance() for s in self.robot.range["right-front"]])
		translation, rotate = self.determineMove(front, left, right)  
		self.robot.move(translation, rotate)
		#print "Sensores 5 y 6 " + str(self.robot.range[5].distance()) + " " + str(self.robot.range[6].distance())
		#print "Sensores 6 y 7 " + str(self.robot.range[6].distance()) + " " + str(self.robot.range[7].distance())

def INIT(engine): 
   assert (engine.robot.requires("range-sensor") and
		   engine.robot.requires("continuous-movement"))
   return Wall_Follower('Wall_Follower', engine) 