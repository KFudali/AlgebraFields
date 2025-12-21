from abc import ABC, abstractmethod
from .stencils import Stencils
import model.domain

class Discretization(ABC):
    def __init__(
        self,
        domain: model.domain.Domain
    ):
        self._domain = domain
    @property
    @abstractmethod
    def dim(self) -> int: pass

    @property
    @abstractmethod
    def stencils(self) -> Stencils: pass
