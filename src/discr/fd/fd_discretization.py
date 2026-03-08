from __future__ import annotations

from typing import Optional
import numpy as np

from discr.core import Discretization
from .operators import FDDiscreteOperatorsFactory
from .bcs import FDDiscreteBCFactory
from .domain import FDDomain

class FDDiscretization(Discretization[FDDomain]):
    def __init__(self, domain: FDDomain):
        self._domain = domain
        self._operators= FDDiscreteOperatorsFactory(domain)
        self._bcs = FDDiscreteBCFactory()

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._domain.grid.shape)

    @property
    def domain(self) -> FDDomain:
        return self._domain

    @property
    def operators(self) -> FDDiscreteOperatorsFactory:
        return self._operators

    @property
    def bcs(self) -> FDDiscreteBCFactory:
        return self._bcs

    def flatten(self, field_array: np.ndarray) -> np.ndarray:
        """Flatten an N-D field array into 1-D ordering used by solvers."""
        return field_array.ravel()

    def reshape(self, field_array: np.ndarray) -> np.ndarray:
        """Reshape a flattened array back into the grid shape."""
        return np.asarray(field_array).reshape(self.shape)