"""
The detector class.
"""

import numpy as np

class Detector:
    """
    The detector class.
    """

    # pictches and ranges of the detector (Constants)
    # There should be nothing that can change this
    _PITCH_X = 0.1
    _PITCH_Y = 0.1
    _RANGE_X = 5
    _RANGE_Y = 5
    # precision
    _EPS = 1e-10

    def __init__(self,
                 center_start  = np.array([ 0, 10, 0]),
                 local_x_start = np.array([-1,  0, 0]),
                 local_y_start = np.array([ 0,  0, 1]),
                 center_curr   = None,
                 local_x_curr  = None,
                 local_y_curr  = None):

        center_start = np.array(center_start, dtype=np.float64)

        # normalize local axes
        local_x_start = np.array(local_x_start, dtype=np.float64)
        local_x_start /= np.linalg.norm(local_x_start)

        local_y_start = np.array(local_y_start, dtype=np.float64)
        local_y_start /= np.linalg.norm(local_y_start)

        # starting center, local coordinates, and normal
        self.center  = {'start': center_start}
        self.local_x = {'start': local_x_start}
        self.local_y = {'start': local_y_start}
        self.normal  = {'start': np.cross(local_x_start, local_y_start)}

        self.__get_curr(center_curr  = center_curr,
                        local_x_curr = local_x_curr,
                        local_y_curr = local_y_curr)


    def __get_curr(self,
                   center_curr  = None,
                   local_x_curr = None,
                   local_y_curr = None):
        """
        add current (misaligned) position if it is given
        """

        center_curr = self.center['start'].copy() \
        if center_curr is None \
            else np.array(center_curr, dtype=np.float64)

        if local_x_curr is not None:
            local_x_curr = np.array(local_x_curr, dtype=np.float64)
            local_x_curr /= np.linalg.norm(local_x_curr)
        else:
            local_x_curr = self.local_x['start'].copy()

        if local_y_curr is not None:
            local_y_curr = np.array(local_y_curr, dtype=np.float64)
            local_y_curr /= np.linalg.norm(local_y_curr)
        else:
            local_y_curr = self.local_y['start'].copy()

        self.center['curr']  = center_curr
        self.local_x['curr'] = local_x_curr
        self.local_y['curr'] = local_y_curr
        self.normal['curr']  = np.cross(local_x_curr, local_y_curr)

    # Alternative initiators
    @classmethod
    def from_dict(cls, parameter_dict):
        """
        Initialize from a parameter dict
        """
        center_start, local_x_start, local_y_start = np.array(parameter_dict['start'],
                                                              dtype=np.float64).reshape(-1, 3)

        center_curr, local_x_curr, local_y_curr = None, None, None
        if 'curr' in parameter_dict:
            center_curr, local_x_curr, local_y_curr = np.array(parameter_dict['curr'],
                                                               dtype=np.float64).reshape(-1, 3)

        return cls(center_start  = center_start,
                   local_x_start = local_x_start,
                   local_y_start = local_y_start,
                   center_curr   = center_curr,
                   local_x_curr  = local_x_curr,
                   local_y_curr  = local_y_curr)

    def get_parameters(self):
        """
        Return the starting and current position of the detector
        """

        return {key: np.hstack([self.center[key],
                                self.local_x[key],
                                self.local_y[key]])
                for key in ['start', 'curr']}

    def __intersect(self, particles, state):
        """
        Calculate intersections of a ray and a detector plane in global coordinates
        Input:
            particles: named tuple with two field vertex and direction
                - vertex: np.array of shape (num_particles, 3)
                - direction: np.array of shape (num_particles, 3)
        """
        # angle between the ray and the detector
        dot_prod = np.dot(particles.direction, self.normal[state])

        # intersection points
        time = np.dot(self.center[state] - particles.vertex, self.normal[state]) / dot_prod
        intersections = particles.vertex + time.reshape(-1, 1) * particles.direction

        # whether the trajectory (a ray) will hit the detector.
        ray_mask = time > 0

        return intersections, ray_mask


    def __to_local_coordinates(self, global_points, state):
        """
        Get the intersection in the detector's local coordinate
        """
        points = global_points - self.center[state]

        local_x = np.dot(points, self.local_x[state])
        local_y = np.dot(points, self.local_y[state])

        local_coords = np.stack([local_x, local_y]).T

        # intersection points within the detector boundary
        in_range_mask = (abs(local_x) < self._RANGE_X) & (abs(local_y) < self._RANGE_Y)

        return local_coords, in_range_mask

    def __bin(self, local_points, rounded):
        """
        bin according to the sensor pitch
        """
        readout = local_points / np.array([self._PITCH_X, self._PITCH_Y])

        if rounded:
            readout = readout.round()

        return readout

    def get_readout(self, particles, state, rounded=False):
        """
        Get detector readout
        """

        global_points, ray_mask = self.__intersect(particles, state)
        local_points, in_range_mask = self.__to_local_coordinates(global_points, state)
        readout = self.__bin(local_points, rounded=rounded)

        # sanity check
        if not rounded:
            diff = np.abs(self.to_global(readout, state) - global_points).max()
            assert diff < self._EPS, \
                ('maximum difference between recovered and ground-truth '
                 f'intersection points ({diff}) exceeds {self._EPS}.')

        # A valid read-out can only be made by a ray that hits
        # the detector in the range of the detector.
        mask = ray_mask & in_range_mask

        return readout, mask

    def to_global(self, readout, state):
        """
        Convert readout to global coordinates.
        readout in shape (N_parts, 2)
        """
        local_points = readout * np.array([self._PITCH_X, self._PITCH_Y])
        local_coord = np.stack([self.local_x[state],
                                self.local_y[state]])
        global_points = np.matmul(local_points, local_coord) + self.center[state]
        return global_points

    # mis-alignments
    def __random_center_shift(self, scale):
        """
        Generate random displacement to the center
        The displacement will be a 3D Normal distribution
        with mean zero and standard deviation by default
        1/100 of the average range of the detector.
        """
        std = scale * (self._RANGE_X + self._RANGE_Y) / 2.
        return np.random.normal(loc=0, scale=std, size=3)

    def shift_center(self, displacement=None, scale=0.01, verbose=True):
        """
        Shift the center of the detector
        and leave local axes (and hence, normal) intact.
        """
        manner = 'specified'
        if displacement is None:
            displacement = self.__random_center_shift(scale=scale)
            manner = 'random'

        self.center['curr'] += np.array(displacement)
        if verbose:
            print(f'Center Shift ({manner}):')
            print(f"\tcenter shift = {np.array2string(displacement, precision=4)}\n")

        return {'displacement': displacement}

    @staticmethod
    def __rodrigues(rotation_axis, rotation_angle, vector):
        """
        Rodrigues' rotation formula
        https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        """
        cos = np.cos(rotation_angle)
        sin = np.sin(rotation_angle)

        cross_prod = np.cross(rotation_axis, vector)
        dot_prod = np.dot(rotation_axis, vector)

        return cos * vector + \
               sin * cross_prod + \
               (1 - cos) * dot_prod * rotation_axis

    def __random_normal_shift(self, scale):
        """
        Sample a direction in a cone of the current normal.
        The direction is generated as follows:
            - first generate a 2D vector following normal
                distribution with standard deviation equaling
                np.sin(scale) / np.sqrt(np.pi / 2).
                The norm of the random vectors generated this way
                follows the Rayleigh distribution with
                standard deviation = np.sin(scale)
            -
        """
        # The sigma in Rayleigh distribution
        # https://en.wikipedia.org/wiki/Rayleigh_distribution
        rayleigh_scale = np.sin(scale) / np.sqrt(np.pi / 2)
        return np.random.normal(loc   = 0,
                                scale = rayleigh_scale,
                                size  = 2)

    def shift_normal(self, displacement=None, scale=np.pi / 36, verbose=True):
        """
        Displacement is a 2D point in the local xy-coordinates.
        """

        # calcluate new normal from displacement
        manner = 'specified'
        if displacement is None:
            displacement = self.__random_normal_shift(scale=scale)
            manner = 'random'

        local_x_coord, local_y_coord = displacement

        global_displacement = local_x_coord * self.local_x['curr'] + \
                              local_y_coord * self.local_y['curr']

        new_normal_direction = np.array(self.normal['curr'] + global_displacement)
        new_normal = new_normal_direction / np.linalg.norm(new_normal_direction)

        if verbose:
            print(f'Normal Shift ({manner}):')
            print(f"\tcurrent normal = {np.array2string(self.normal['curr'], precision=4)}")

        # Rodrigues' rotation formula
        rotation_axis = np.cross(self.normal['curr'], new_normal)
        rotation_axis /= np.linalg.norm(rotation_axis)

        rotation_angle = np.arccos(np.dot(self.normal['curr'], new_normal))

        # rotate local axes together with normal
        self.local_x['curr'] = self.__rodrigues(rotation_axis,
                                                rotation_angle,
                                                self.local_x['curr'])
        self.local_y['curr'] = self.__rodrigues(rotation_axis,
                                                rotation_angle,
                                                self.local_y['curr'])

        self.normal['curr'] = np.cross(self.local_x['curr'], self.local_y['curr'])

        # check the normal calculated from the new local axes is the same
        # as the new normal
        error = np.linalg.norm(self.normal['curr'] - new_normal)

        assert error < self._EPS

        if verbose:
            print(f"\tnew normal = {np.array2string(new_normal, precision=4)}")
            print(f"\tangle between two normals: {rotation_angle * 180 / np.pi:.2f}\u00B0")
            print(f"\tnew local x-axis: {np.array2string(self.local_x['curr'], precision=4)}")
            print(f"\tnew local y-axis: {np.array2string(self.local_y['curr'], precision=4)}\n")

        return {'displacement': displacement}

    def rotate_axes(self, phi=None, scale=np.pi / 36, verbose=True):
        """
        Rotate the local x and y-axes by phi on the current plane of
        the detector. The normal and the center of the detector are intact.
        """
        manner = 'specified'
        if phi is None:
            phi = np.random.normal(loc=0, scale=scale)
            manner = 'random'

        rotation_mat = np.array([[ np.cos(phi), np.sin(phi)],
                                 [-np.sin(phi), np.cos(phi)]])
        axes = np.stack([self.local_x['curr'], self.local_y['curr']])
        self.local_x['curr'], \
        self.local_y['curr'] = np.matmul(rotation_mat, axes)

        # check wither the rotated x, y-axes are still perpendicular
        error = np.dot(self.local_x['curr'], self.local_y['curr'])
        assert error < self._EPS

        if verbose:
            print(f'Rotate Local Axes ({manner}):')
            print(f"\trotaion angle = {phi * 180 / np.pi:.2f}\u00B0")
            print(f"\tnew local x-axis: {np.array2string(self.local_x['curr'], precision=4)}")
            print(f"\tnew local y-axis: {np.array2string(self.local_y['curr'], precision=4)}\n")

        return {'phi': phi}

    def misalign(self, misalign_type, kwargs, verbose=True):
        """
        Misalign by type and keyword arguments
        """

        if misalign_type == 'center_shift':
            return self.shift_center(**kwargs, verbose=verbose)

        if misalign_type == 'normal_shift':
            return self.shift_normal(**kwargs, verbose=verbose)

        if misalign_type == 'axes_rotation':
            return self.rotate_axes(**kwargs, verbose=verbose)

        raise ValueError(f'Unknown misalign type {misalign_type}')
