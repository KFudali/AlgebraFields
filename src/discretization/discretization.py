from abc import ABC, abstractmethod
from domain import Domain
from .stencils import Stencils

class Discretization(ABC):
    def __init__(
        self,
        domain: Domain
    ):
        self._domain = domain

    @property
    @abstractmethod
    def stencils(self) -> Stencils: pass
