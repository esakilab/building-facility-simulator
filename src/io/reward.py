from typing import NamedTuple, Type, TypeVar

from src.io.building_state import AreaState

T = TypeVar('T', bound='Reward')

class Reward(NamedTuple):
    metric1: float

    @classmethod
    def from_area_states(cls: Type[T], area_states: list[AreaState]) -> T:
        return cls(
            metric1=0
        )