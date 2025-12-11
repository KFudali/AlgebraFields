import numpy as np
import enum
from typing import Optional

class Dim(enum.IntEnum):
    D1 = 1
    D2 = 2
    D3 = 3

class MeshInitException(Exception): pass

class ElementMesh():
    def __init__(
        self,
        dim: Dim,
        points: np.ndarray,
        connectivity: np.ndarray,
        conn_offsets: np.ndarray
    ):
        self._dim = dim
        self._assign_points(points)
        self._assign_connectivity(connectivity, conn_offsets)

    @property
    def dim(self) -> Dim: return self._dim
    
    @property
    def n_points(self) -> int: return self._points.shape[0]

    def _assign_points(self, points: np.ndarray):
        if points.ndim == 1:
            if points.size % self.dim.value != 0:
                raise MeshInitException(
                    f"Number of entries in points ({points.size}) "
                    f"is not divisible by self.dimension {self.dim.value}"
                )
            n_points = points.size // self.dim.value
            points = points.reshape((n_points, self.dim.value))
        if points.ndim == 2:
            if points.shape[1] != self.dim.value:
                raise MeshInitException(
                    f"Points array has shape {points.shape}, "
                    f"but self.dimension is {self.dim.value}"
                )
        self._points = points

    def _assign_connectivity(
        self, 
        conn: np.ndarray,
        conn_offsets: np.ndarray
    ):
        self._conn = conn
        self._conn_offsets = conn_offsets


class ElementMesh1D(ElementMesh):
    def __init__(
        self,
        points: np.ndarray,
        connectivity: np.ndarray,
        conn_offsets: np.ndarray 
    ): pass

class ElementMesh2D(ElementMesh):
    def __init__(
        self,
        points: np.ndarray,
        connectivity: np.ndarray,
        conn_offsets: np.ndarray 
    ): pass

class ElementMesh3D(ElementMesh):
    def __init__(
        self,
        points: np.ndarray,
        connectivity: np.ndarray,
        elem_types: np.ndarray,
        conn_offsets: np.ndarray 
    ): pass