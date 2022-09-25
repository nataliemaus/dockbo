"""Module for representing dimension boundaries"""
import numpy as np 
from .lightdock_constants import (
    MAX_TRANSLATION,
    MAX_ROTATION,
    MIN_EXTENT,
    MAX_EXTENT
)

class Boundary(object):
    """Represents a boundary for a given dimension"""

    def __init__(self, lower_limit, upper_limit):
        """Creates a boundary of a dimension with lower and upper limits"""
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    def clone(self):
        """Gets a copy of this boundary"""
        return Boundary(self.lower_limit, self.upper_limit)

    def __eq__(self, other):
        """Compares for equality two boundaries"""
        lowers_eq = np.isclose(self.lower_limit,other.lower_limit )
        uppers_eq =  np.isclose(self.upper_limit,other.upper_limit )
        return lowers_eq == uppers_eq

    def __ne__(self, other):
        """Compares for unequality two boundaries"""
        return not self.__eq__(other)

    def __repr__(self):
        return "[%s, %s]" % (self.lower_limit, self.upper_limit)


class BoundingBox(object):
    """Represents a set of boundaries to apply to each dimension of a given space"""

    def __init__(self, boundaries):
        """Creates a bounding box with a Boundary for each dimension"""
        self.boundaries = boundaries
        self.dimension = len(self.boundaries)

    def get_boundary_of_dimension(self, dimension_index):
        """Gets the Boundary of the dimension with dimension_index"""
        return self.boundaries[dimension_index]

    def __repr__(self):
        return " ".join([str(b) for b in self.boundaries])


# lightdock code: 
def get_default_box(use_anm, anm_rec, anm_lig):
    """Get the default bounding box"""
    boundaries = [
        Boundary(-MAX_TRANSLATION, MAX_TRANSLATION),
        Boundary(-MAX_TRANSLATION, MAX_TRANSLATION),
        Boundary(-MAX_TRANSLATION, MAX_TRANSLATION),
        Boundary(-MAX_ROTATION, MAX_ROTATION),
        Boundary(-MAX_ROTATION, MAX_ROTATION),
        Boundary(-MAX_ROTATION, MAX_ROTATION),
        Boundary(-MAX_ROTATION, MAX_ROTATION),
    ]
    if use_anm:
        boundaries.extend([Boundary(MIN_EXTENT, MAX_EXTENT) for _ in range(anm_rec)])
        boundaries.extend([Boundary(MIN_EXTENT, MAX_EXTENT) for _ in range(anm_lig)])

    return BoundingBox(boundaries)