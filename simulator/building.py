from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from simulator.area import Area, AreaAction, AreaState
from simulator.environment import ExternalEnvironment


class BuildingState(NamedTuple):
    """ビル設備全体の状態を表すオブジェクト
    2.5節 AIがアクセスできるもの に対応
    """
    NDARRAY_ELEMS = 4

    areas: list[AreaState]
    power_balance: float
    electric_price_unit: float
    solar_radiation: float
    temperature: float
    


    def to_ndarray(self) -> np.ndarray:
        area_states_ndarray = np.concatenate([
            area_state.to_ndarray() for area_state in self.areas
        ])

        return np.concatenate([
            area_states_ndarray,
            np.array([
                self.power_balance, 
                self.electric_price_unit, 
                self.solar_radiation, 
                self.temperature])
        ])


    @classmethod
    def create(cls, area_states: list[AreaState], ext_env: ExternalEnvironment) -> BuildingState:
        return cls(
            areas=area_states,
            power_balance=sum(state.power_consumption for state in area_states),
            electric_price_unit=ext_env.electric_price_unit,
            solar_radiation=ext_env.solar_radiation,
            temperature=ext_env.temperature
        )


@dataclass
class BuildingAction():
    """AIからの指示を表すオブジェクト
    """

    areas: list[AreaAction]

    def from_ndarray(src: np.ndarray, areas: list[Area]) -> BuildingAction:
        area_actions = []
        for area in areas:
            area_action = AreaAction.from_ndarray(src, area.facilities)
            
            area_actions.append(area_action)
            src = src[area_action.consumed_ndarray_len:]
        
        return BuildingAction(areas=area_actions)
