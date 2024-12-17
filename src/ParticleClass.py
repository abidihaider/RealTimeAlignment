from src.TimeClass import TimeClass
import numpy as np

class ParticleClass(TimeClass):
    def __init__(self):
        super().__init__()

        # starting position
        self.startPos = np.array([0, 0, 0])
        self.startDir = np.array([0, 1, 0])


    # the start position and direction are np array of size 3
    # that specific the starting position and direction
    def setStartPosAndDir(self, position, direction):
        self.startPos = position
        self.startDir = direction;


    # getter for current position
    def getCurrPos(self):
        return self.startPos

    # getter for current direction
    def getCurrDir(self):
        return self.startDir
