import numpy as np
from ..core import SubDomain, SubDomainId

class FDSubDomain(SubDomain):
    def __init__(
        self,
        subdomain_id: SubDomainId,
        ids: np.ndarray
    ):
        super().__init__(subdomain_id)
        self._ids = ids

    @property
    def ids(self) -> np.ndarray: return self._ids
