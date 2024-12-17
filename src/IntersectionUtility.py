#!/bin/python
from __future__ import print_function
import numpy as np


def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):

    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi
