from typing import NamedTuple, Type, TypeVar

from src.io.state import AreaState

T = TypeVar('T', bound='Reward')

class Reward(NamedTuple):
    """報酬を表すタプルオブジェクト
    """

    metric1: float

    @classmethod
    def from_area_states(cls: Type[T], area_states: list[AreaState]) -> T:
        return cls(
            metric1=0
        )