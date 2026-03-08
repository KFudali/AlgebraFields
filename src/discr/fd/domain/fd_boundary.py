from dataclasses import dataclass
from discr.core.domain import Boundary
from tools.region import Region
from tools.geometry import StructuredGridND
@dataclass(frozen=True)
class FDBoundary(Boundary):
    region: Region
    axis: int
    inward_dir: int
    grid: StructuredGridND