import numpy as np
from shapely.geometry import Point

class Distance:

    def __init__(self, meters):
        self._distance = meters

    def destination(self, center, bearing):
        """
        Compute the destination from center and direction angle (bearing)

        Parameters
        ----------
            center : (y,x) tuple. Coordinates are y,x to ensure harmonisation with geopy functions where coordinates are expressed in (lat,lon)
            bearing : direction angle in degrees

        Return
        ------
            point : destination point
        """

        angle = bearing * np.pi / 180.0
        nx = np.sin(angle)
        ny = np.cos(angle)

        dest_x = center[1] + nx * self._distance
        dest_y = center[0] + ny * self._distance

        return Point(dest_x, dest_y)
    
distance = Distance