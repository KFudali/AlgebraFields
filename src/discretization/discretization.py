from abc import ABC, abstractmethod
from domain import Domain

class Discretization(ABC):
    def __init__(
        self,
        domain: Domain
    ):
        self._domain = domain
