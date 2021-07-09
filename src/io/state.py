from abc import ABC
from dataclasses import dataclass
from typing import NamedTuple, Type, TypeVar


T = TypeVar('T', bound='BuildingState')


class FacilityState(ABC):
    """AIに出力する、設備の状態を表す抽象クラス
    """
    pass

    @classmethod
    def empty(cls):
        return cls()


class AreaState(NamedTuple):
    """エリアの状態を表すオブジェクト
    """

    power_consumption: float
    temperature: float
    people: int
    facilities: FacilityState


class BuildingState(NamedTuple):
    """ビル設備全体の状態を表すオブジェクト
    2.5節 AIがアクセスできるもの に対応
    """

    areas: list[AreaState]
    power_balance: float
    electric_price_unit: float

    @classmethod
    def create(cls: Type[T], area_states: list[AreaState], electric_price_unit: float) -> T:
        return cls(
            areas=area_states,
            power_balance=sum(state.power_consumption for state in area_states),
            electric_price_unit=electric_price_unit
        )
