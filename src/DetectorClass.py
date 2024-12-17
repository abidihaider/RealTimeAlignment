from src.TimeClass import TimeClass
from src.IntersectionUtility import LinePlaneCollision

import numpy as np
import math

class DetectorClass(TimeClass):
    def __init__(self):
        super().__init__()

        # starting position
        self.startPos = np.array([0, 0, 0])
        # starting normal for the sensor plane
        self.startNor = np.array([0, 1, 0])

        # rotation around the normal
        self.rotAroundNor = 0

        # current position - This is include any time dependant update we want to make
        # but at the start they are same as starting position
        self.currPos = self.startPos
        self.currNor = self.startNor
        self.currRotAroundNor = self.rotAroundNor;

        self.pitchX = 0.1
        self.pitchY = 0.1

        self.misalignmentFuncList = []


    # update the position of the detector based on the current time
    # Todo: make it take lambda that updates the position
    def updateDetectorAlignment(self):
        if(len(self.misalignmentFuncList) == 0):
            return

        # if there are any function, apply all the function as given
        currP = self.getCurrPos()
        currN = self.getCurrNor()
        currR = self.getCurrRot()

        for _func in self.misalignmentFuncList:
            currP, currN, currR = _func(self.time, currP, currN, currR)    


        # update the position
        self.currPos = currP
        self.currNor = currN
        self.currRotAroundNor = currR

    # the start position and normal are np array of size 3
    # that specific the starting position and startNor
    def setStartPosAndNor(self, position, normal, rotation):
        self.startPos = position
        self.startNor = normal;
        self.rotAroundNor = rotation

        self.currPos = self.startPos
        self.currNor = self.startNor
        self.currRotAroundNor = self.rotAroundNor;

    # Add to the list on how misalignment will be update
    def addMisalignmentFunction(self, _func):
        self.misalignmentFuncList.append(_func)

    # getter for current position
    def getCurrPos(self):
        return self.currPos

    # getter for current normal
    def getCurrNor(self):
        return self.currNor

    # getter for current rotation
    def getCurrRot(self):
        return self.currRotAroundNor

    # calculate intersection in global coordinates
    def getTruthGlobalIntersectionPosition(self, particle):
        intersectionPoint = LinePlaneCollision(planeNormal = self.getCurrNor(), planePoint = self.getCurrPos(), rayDirection = particle.getCurrDir(), rayPoint = particle.getCurrPos())
        return intersectionPoint


    # get the true intersection position
    def getLocalPositionFromGlobal(self, globalPoint):
        localPos = globalPoint - self.getCurrPos()

        # Z axis should always be zero
        assert localPos[2] == 0

        # rotate the position along the local axis
        cos_angle = math.cos(self.getCurrRot())
        sin_angle = math.sin(self.getCurrRot())
        x_ = localPos[0] * cos_angle - localPos[1] * sin_angle
        y_ = localPos[0] * sin_angle + localPos[1] * cos_angle

        return np.array([x_, y_])

    # bin position according to sensor pitch
    def getDetectorHitLocationFromLocal(self, localPosition):
        assert len(localPosition) == 2

        xBin = round(localPosition[0] / self.pitchX)
        yBin = round(localPosition[1] / self.pitchY)

        return np.array([xBin, yBin])

    # reco the detector level position - includes noise and reconstruction error
    def getRecoPointFromLocalPosition(self, localTruthPoint):

        # right now return just the truth position, but we can add sources of error here
        return localTruthPoint


