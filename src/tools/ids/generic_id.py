from dataclasses import dataclass
from typing import TypeVar, Type, ClassVar, Generic, Any

K = TypeVar("K")
T = TypeVar("T", bound="GenericId[Any]")

@dataclass(frozen=True)
class GenericId(Generic[K]):
    """Generic id parametrised by key type K. Subclasses only need to set INVALID_KEY."""
    _key: K
    INVALID_KEY: ClassVar[K] = None

    @property
    def key(self) -> K:
        return self._key

    @property
    def is_valid(self) -> bool:
        return self._key != self.INVALID_KEY

    @classmethod
    def invalid(cls: Type[T]) -> T:
        return cls(cls.INVALID_KEY)

@dataclass(frozen=True)
class IntId(GenericId[int]):
    INVALID_KEY: ClassVar[int] = -1