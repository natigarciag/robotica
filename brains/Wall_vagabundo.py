# robot goes forward and then slows to a stop when it detects something  
	
from pyrobot.brain import Brain
import numpy as np

def between(number, point1, point2):
	if number > point1 and number < point2:
		return 0
	elif number < point1:
		return -1
	else:
		return 1

class WallFollower(Brain):

	# Give the front two sensors, decide the next move  
	def determineMove(self, front, left, right):
		try:
			previousMeasurements = self.previousMeasurements
		except AttributeError as ae:
			self.previousMeasurements = np.array([[self.robot.range[i].distance() for i in range(8)] for j in range(5)])
			previousMeasurements = self.previousMeasurements

		try:
			state = self.state
		except AttributeError as ae:
			self.state = 'initial'
			state = self.state
		
		measurements = np.array([[self.robot.range[i].distance() for i in range(8)] for j in range(5)])
		self.previousMeasurements = measurements

		meanMeasurements = np.mean(measurements, axis=0)
		print meanMeasurements

		translation = 0
		rotation = 0


		if self.state == 'initial':
			if (front < 0.3 or right < 0.1 or left < 0.1):
				self.state = 'rotateToFindWall'
				# doesn't set translation nor rotation. This means to stop
			else:
				translation = 0.5
				rotation = 0
		
		elif (self.state == 'rotateToFindWall'):
			if meanMeasurements[3] < meanMeasurements[4] or meanMeasurements[6] - 0.1 < meanMeasurements[7]:
				# rotate to the left
				rotation = 0.1
			else:
				self.state = 'followWall'
		
		elif self.state == 'followWall':
			rightiest = 7
			leftiest = 5
			sevenTooNear = 0.2
			sevenTooFar = 0.3
			sixTooNear = 0.27
			sixTooFar = 0.37

			if front < 0.5 and meanMeasurements[3] < 0.6:
				self.state = 'rotateNextWallLeft'
			elif between(meanMeasurements[rightiest], sevenTooNear, sevenTooFar) == -1 and between(meanMeasurements[leftiest], sixTooNear, sixTooFar) == -1:
				# get farther from wall fast
				translation = 0.7
				rotation = -0.2
			elif between(meanMeasurements[rightiest], sevenTooNear, sevenTooFar) == -1 and between(meanMeasurements[leftiest], sixTooNear, sixTooFar) == 0:
				# get farther from wall, but it's good
				translation = 1
				rotation = -0.1
			elif between(meanMeasurements[rightiest], sevenTooNear, sevenTooFar) == -1 and between(meanMeasurements[leftiest], sixTooNear, sixTooFar) == 1:
				# getting farther from wall
				translation = 1

			elif between(meanMeasurements[rightiest], sevenTooNear, sevenTooFar) == 0 and between(meanMeasurements[leftiest], sixTooNear, sixTooFar) == -1:
				# getting near the wall
				translation = 0.6
				rotation = -0.6
			elif between(meanMeasurements[rightiest], sevenTooNear, sevenTooFar) == 0 and between(meanMeasurements[leftiest], sixTooNear, sixTooFar) == 0:
				# perfect
				translation = 1
			elif between(meanMeasurements[rightiest], sevenTooNear, sevenTooFar) == 0 and between(meanMeasurements[leftiest], sixTooNear, sixTooFar) == 1:
				# getting farther. Compensate
				translation = 1
				rotation = 0.1

			elif between(meanMeasurements[rightiest], sevenTooNear, sevenTooFar) == 1 and between(meanMeasurements[leftiest], sixTooNear, sixTooFar) == -1:
				# getting perilously near the wall
				translation = 1
				rotation = 0.5
			elif between(meanMeasurements[rightiest], sevenTooNear, sevenTooFar) == 1 and between(meanMeasurements[leftiest], sixTooNear, sixTooFar) == 0:
				# getting perilously near the wall
				translation = 1
				rotation = 0.5
			elif between(meanMeasurements[rightiest], sevenTooNear, sevenTooFar) == 1 and between(meanMeasurements[leftiest], sixTooNear, sixTooFar) == 1:
				# far from the wall
				rotation = -0.2
				translation = 0.6


			# elif meanMeasurements[7] > 0.25 and meanMeasurements[6] > 0.32:
			# 	translation = 0.6
			# 	rotation = -0.2
			# elif meanMeasurements[7] < 0.2 and meanMeasurements[6] < 0.27:
			# 	translation = 0.6
			# 	rotation = 0.2
			else:
				# translation = 0.8
				# rotation = 0
				print 'this case isnt covered'
		
		elif self.state == 'rotateNextWallLeft':
			if meanMeasurements[3] < meanMeasurements[4] or meanMeasurements[6] - 0.1 < meanMeasurements[7]:
				# rotate to the left
				rotation = 0.2
			else:
				self.state = 'followWall'


		action = (translation, rotation)
		print str(self.state) + ' ' + str(action)
		return action

		# if front < 0.5:   
		# 	print "obstacle ahead, hard turn"  
		# 	return(0, .3)  
		# elif left < 0.8:
		# 	print "object detected on left, slow turn"
		# 	return(0.1, -.3)  
		# elif right < 0.8: 
		# 	print "object detected on right, slow turn" 
		# 	return(0.1, .3)  
		# else:  
		# 	print "clear"  
		# 	return(0.5, 0.0) 
		
		
	def step(self):  
		front = min([s.distance() for s in self.robot.range["front"]])
		left = min([s.distance() for s in self.robot.range["left-front"]])
		right = min([s.distance() for s in self.robot.range["right-front"]])
		translation, rotate = self.determineMove(front, left, right)  
		self.robot.move(translation, rotate)

def INIT(engine):  
	assert (engine.robot.requires("range-sensor") and
			  engine.robot.requires("continuous-movement"))
	return WallFollower('WallFollower', engine)

