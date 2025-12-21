from ..discretization import Discretization
from .domain import FDDomain

class FDDiscretization(Discretization):
    def __init__(self, domain: FDDomain):
        super().__init__(domain)