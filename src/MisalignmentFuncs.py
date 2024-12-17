import numpy as np
import math


# return things as is
def noChange(time, currPosition, currNormal, currRotation):
    return currPosition, currNormal, currRotation


def slowYShift(time, currPosition, currNormal, currRotation):
    pos = currPosition + np.array([0.0, 0.1, 0])
    return pos, currNormal, currRotation


def slowRotate(time, currPosition, currNormal, currRotation):
    rotate = currRotation + 0.01
    return currPosition, currNormal, rotate