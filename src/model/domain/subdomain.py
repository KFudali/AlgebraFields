from .subdomain_id import SubDomainId

class SubDomain():
    def __init__(
        self,
        subdomain_id: SubDomainId
    ):
        self._id = subdomain_id

    @property
    def id(self) -> SubDomainId: return self._id