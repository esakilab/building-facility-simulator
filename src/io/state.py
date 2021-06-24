from typing import NamedTuple, Type, TypeVar


T = TypeVar('T', bound='BuildingState')


class AreaState(NamedTuple):
    """エリアの状態を表すオブジェクト
    """

    power_consumption: float
    temperature: float
    people: int


class BuildingState(NamedTuple):
    """ビル設備全体の状態を表すオブジェクト
    2.5節 AIがアクセスできるもの に対応
    """

    area_states: list[AreaState]
    power_balance: float

    @classmethod
    def from_area_states(cls: Type[T], area_states: list[AreaState]) -> T:
        return cls(
            area_states=area_states,
            power_balance=sum(state.power_consumption for state in area_states)
        )
