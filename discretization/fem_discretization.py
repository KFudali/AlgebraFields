from .abstract_discretization import AbstractDiscretization
from meshutils.connectivity import Connectivity3D

class FemDiscretization(AbstractDiscretization):
    def __init__( self, connectivity: Connectivity3D):
        super().__init__()








# Suppose i am solving N-S equations
# 