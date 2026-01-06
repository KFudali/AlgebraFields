from abc import ABC, abstractmethod

from .subdomain_id import SubDomainId
from .subdomain import SubDomain

from .boundary_id import BoundaryId
from .boundary import Boundary

class Domain(ABC):
    @property
    @abstractmethod
    def subdomains(self) -> list[SubDomainId]: pass

    @property
    @abstractmethod
    def boundaries(self) -> list[BoundaryId]: pass

    @abstractmethod
    def subdomain(self, subdomain_id: SubDomainId) -> SubDomain: pass
    
    @abstractmethod
    def boundary(self, boundary_id: BoundaryId) -> Boundary: pass