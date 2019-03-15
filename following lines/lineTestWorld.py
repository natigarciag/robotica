"""
A simulation with a simple line on the floor.
"""

from lineSimulation import LineSimulation
from pyrobot.simulators.pysim import TkPioneer, \
     PioneerFrontSonars, PioneerFrontLightSensors

def INIT():
    # (width, height), (offset x, offset y), scale
    sim = LineSimulation((450,675), (20,650), 32,
                            #  background="line-images/lineBackground-1.png")  
                            #  background="line-images/lineBackground-2.png")  
                            background="line-images/IMG_0037.png")
                            # background="line-images/IMG_0038.png")

    # an example of an obstacle on the line
      # x1, y1, x2, y2

    # # sim.addBox(5, 12, 6, 11)
    # # sim.addBox(5, 11.7, 6, 10.7)
    # # sim.addBox(4, 11, 5, 10)
    # # sim.addBox(4, 10, 5, 9)

    # # sim.addBox(4, 3.5, 5, 4.5)
    # # # sim.addBox(4, 2.5, 5, 3.5)
    # # # sim.addBox(3.7, 2.5, 4.7, 3.5)
    # sim.addBox(13.4, 12.6, 14.4, 13.6)

    sim.addRobot(60000, 
		 # name, x, y, th, boundingBox
                 TkPioneer("RedErratic", 
			   # position for lineBackground-1
		           1, 18.9, 4.0,
			   # position for lineBackground-2
		           # 8.5, 2.35, 1.57,
                           ((.185, .185, -.185, -.185),
                            (.2, -.2, -.2, .2))))

    # add some sensors:
    sim.robots[0].addDevice(PioneerFrontSonars())

    # to create a trail
    sim.robots[0].display["trail"] = 1

    return sim
