"""
The particle class
"""

from collections import namedtuple
import numpy as np


Particles = namedtuple('Particles', ['vertex', 'direction'])

class RandomParticle:
    """
    The particle class that can generate random particles
    """
    def __init__(self,
                 vertex_mean    = np.array([0, 0, 0]),
                 vertex_std     = np.array([.1, .1, .1]),
                 direction_mean = np.array([0, 1, 0]),
                 direction_std  = np.array([.1, .1, .1])):

        self.vertex_mean    = np.array(vertex_mean,    dtype=np.float64)
        self.vertex_std     = np.array(vertex_std,     dtype=np.float64)
        self.direction_mean = np.array(direction_mean, dtype=np.float64)
        self.direction_std  = np.array(direction_std,  dtype=np.float64)

    def __call__(self, num_particles):
        """
        generate particles with random vertex and direction
        """
        vertices = np.random.normal(loc   = self.vertex_mean,
                                    scale = self.vertex_std,
                                    size  = (num_particles, 3))

        directions = np.random.normal(loc   = self.direction_mean,
                                      scale = self.direction_std,
                                      size  = (num_particles, 3))

        return Particles(vertex=vertices, direction=directions)
