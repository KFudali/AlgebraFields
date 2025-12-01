import numpy as np
import numpy.typing as npt

from .elem_type import ElemType

class Connectivity2D():
    @classmethod
    def from_vertices_and_offsets(
        element_vertices: npt.NDArray[np.int_], 
        offsets: np.ndarray[np.int_]
    ): pass

class Connectivity3D():
    @classmethod
    def from_vertices_and_offsets(
        element_vertices: npt.NDArray[np.int_], 
        offsets: npt.NDArray[np.int_],
        element_types: npt.NDArray[np.int_]
    ): pass
