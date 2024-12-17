#!/usr/bin/env python
#
# This script submits jobs to the grid
import os

from src.TimeClass import TimeClass
from src.DetectorClass import DetectorClass
from src.ParticleClass import ParticleClass
from src.IntersectionUtility import LinePlaneCollision
import src.MisalignmentFuncs as MisalignmentFuncs

import numpy as np
import json
import random 


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--outputName",  type=str, required=True, help="name of the output file")
parser.add_argument("--nEvents",  type=int, required=True, help="number of tracks wanted")
args = parser.parse_args()


def createDetectorList():
    detector1 = DetectorClass()
    detector1.setStartPosAndNor(position = np.array([0, 10, 0]), normal = np.array([0, 1, 0]), rotation = 0)

    detector2 = DetectorClass()
    detector2.setStartPosAndNor(position = np.array([0, 20, 0]), normal = np.array([0, 1, 0]), rotation = 0)
    detector2.addMisalignmentFunction(MisalignmentFuncs.slowYShift)

    detector3 = DetectorClass()
    detector3.setStartPosAndNor(position = np.array([0, 30, 0]), normal = np.array([0, 1, 0]), rotation = 0)

    detectorList = []
    detectorList.append(detector1)
    detectorList.append(detector2)
    detectorList.append(detector3)

    return detectorList;


def plotEvent(detectorList, particle):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt3d = plt.figure().add_subplot(projection = '3d')


    globalPoints = []
    globalPoints.append(particle.getCurrPos())
    for detector in detectorList:
        point  = detector.getCurrPos()
        normal = detector.getCurrNor()

        # a plane is a*x+b*y+c*z+d=0
        # [a,b,c] is the normal. Thus, we have to calculate
        # d and we're set
        d = -point.dot(normal)

        # create x,y
        xx, zz = np.meshgrid(range(-10, 10), range(-10, 10))

        # calculate corresponding z
        yy = (-normal[0] * xx - normal[2] * zz - d) * 1. / normal[1]

        # plot the surface
        plt3d.plot_surface(xx, yy, zz, alpha=0.2)

        globalPoints.append(detector.getTruthGlobalIntersectionPosition(particle))

    globalPoints.append(detector.getTruthGlobalIntersectionPosition(particle))

    #Plot the intersection : 
    # for points in globalPoints:
        # plt3d.scatter(points[0] , points[1] , points[2],  color='green')

    for i in range(0, len(globalPoints) - 1):
        pos1 = globalPoints[i]
        pos2 = globalPoints[i+1]
        plt3d.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],zs=[pos1[2], pos2[2]],  color='green', marker='o')


    # plot the starting point
    pos = particle.getCurrPos()
    plt3d.scatter(pos[0] , pos[1] , pos[2],  color='red')


    plt3d.set_xlabel('$X$')
    plt3d.set_ylabel('$Y$')
    plt3d.set_zlabel('$Z$')

    plt.show()


def main():
    # set random number seed for now
    random.seed(10)

    # create time class for incrementing time
    classTime = TimeClass()
    detectorList = createDetectorList()

    ## for plotting
    # particle = ParticleClass()
    # particle.setStartPosAndDir(position = np.array([0, 0, 0],), direction = np.array([0.1, 1, 0]))

    # plotEvent(detectorList, particle)

    # number of particles
    nParticles = 0

    # output data dictionary
    outputData = {}

    # particle loop
    while(nParticles < args.nEvents):

        outputEvntData = {}

        # generate a particle with some random properties
        particle = ParticleClass()
        particle.setStartPosAndDir(position = np.array([random.gauss(0, 10), 0, 0],), direction = np.array([random.gauss(0, 1), random.gauss(1, 0.1), 0]))

        # store the true particle
        outputEvntData["particle"] = {};
        outputEvntData["particle"]["position"]  = particle.getCurrPos().tolist()
        outputEvntData["particle"]["direction"] = particle.getCurrDir().tolist()


        detectorIndex = 0
        for detector in detectorList:
            globalTruthPoint = detector.getTruthGlobalIntersectionPosition(particle)
            localTruthPoint = detector.getLocalPositionFromGlobal(globalPoint = globalTruthPoint)
            localRecoPoint = detector.getRecoPointFromLocalPosition(localTruthPoint = localTruthPoint)
            localRecoBin = detector.getDetectorHitLocationFromLocal(localPosition = localRecoPoint)

            key = "detector_" + str(detectorIndex)
            detectorIndex += 1
            outputEvntData[key] = {}
            outputEvntData[key]["localRecoHitBin"]  = localRecoBin.tolist()
            outputEvntData[key]["localRecoPoint"]   = localRecoPoint.tolist()
            outputEvntData[key]["localTruthPoint"]  = localTruthPoint.tolist()
            outputEvntData[key]["globalTruthPoint"] = globalTruthPoint.tolist()
            outputEvntData[key]["truthDetectorPos"] = detector.getCurrPos().tolist()
            outputEvntData[key]["truthDetectorNor"] = detector.getCurrNor().tolist()
            outputEvntData[key]["truthDetectorRotat"] = detector.getCurrRot()

            # print (globalTruthPoint, localTruthPoint, localRecoBin)


        classTime.incrementTime()
        for detector in detectorList:
            detector.updateDetectorAlignment()


        outputData["Event"+str(nParticles)] = outputEvntData
        nParticles += 1

    # print(outputData)
    with open(args.outputName, 'w') as f:
        json.dump(outputData, f)

    import pprint
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(outputData)


# run the main function
if __name__ == "__main__":
    main()

