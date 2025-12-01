import numpy as np
import numpy.typing as npt
from  .connectivity import Connectivity3D


class Mesh():
    def __init__(
        self,
        points: npt.NDArray[np.int_],
        connectivity: Connectivity3D
    ):
        pass