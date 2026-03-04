from dataclasses import dataclass
from discretization.core.domain import Boundary
from tools.region import Region

@dataclass(frozen=True)
class FDBoundary(Boundary):
    region: Region
    axis: int
    inward_dir: int
