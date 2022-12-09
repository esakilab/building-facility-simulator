from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple, Optional, TypeVar

import numpy as np

from simulator.facility import Facility
from simulator.environment import ExternalEnvironment, AreaEnvironment
from simulator.facility.facility_base import FacilityAction, FacilityActionFactory, FacilityState


ALPHA = 0.02

T = TypeVar('T', bound='Area')

@dataclass
class Area:
    """エリアを表すオブジェクト
    """

    name: str
    facilities: list[Facility]
    simulate_temperature: bool
    capacity: float
    temperature: float
    power_consumption: float = 0
    people: int = 0
    

    def update(
        self, 
        action: AreaAction,
        ext_env: ExternalEnvironment, 
        area_env: Optional[AreaEnvironment] = None,
    ) -> AreaState:
        """ext_envとarea_envに応じて温度と消費電力を更新する

        2.6節の2,3に対応
        """
        self.people = area_env.people if area_env else 0

        beta = area_env.calc_beta() if area_env else 0
        self.power_consumption = 0.

        for fid, facility in enumerate(self.facilities):
            state, effect = facility.update(
                action=action.facilities[fid],
                ext_env=ext_env,
                area_temperature=self.temperature)

            beta += effect.heat * 60
            self.power_consumption += effect.power

        if self.simulate_temperature:
            assert area_env
            
            temp_dif = self.temperature - ext_env.temperature
            self.temperature += -ALPHA * temp_dif + beta / (self.capacity * 1.189)

        else:
            self.temperature = ext_env.temperature

        return self.get_state()
    

    def get_state(self) -> AreaState:
        return AreaState(
            power_consumption=self.power_consumption,
            temperature=self.temperature,
            people=self.people,
            facilities=[facility.get_state() for facility in self.facilities]
        )
    

    def get_state_shape(self) -> tuple[int]:
        return (AreaState.NDARRAY_ELEMS + sum(f.STATE_TYPE.NDARRAY_SHAPE[0] for f in self.facilities),)


class AreaState(NamedTuple):
    """エリアの状態を表すオブジェクト
    """
    NDARRAY_ELEMS = 3

    power_consumption: float
    temperature: float
    people: int
    facilities: list[FacilityState]


    def to_ndarray(self) -> np.ndarray:
        facility_states_ndarray = np.concatenate([
            facility_state.to_ndarray() for facility_state in self.facilities
        ])

        return np.concatenate([
            facility_states_ndarray,
            np.array([
                self.power_consumption, 
                self.temperature, 
                float(self.people)])
        ])


@dataclass
class AreaAction:
    facilities: list[FacilityAction]
    consumed_ndarray_len: int

    def from_ndarray(src: np.ndarray, facilities: list[Facility]) -> AreaAction:
        facility_actions = []
        cursor = 0
        for facility in facilities:
            next_cursor = cursor + facility.ACTION_TYPE.NDARRAY_SHAPE[0]
            cut = src[cursor:next_cursor]
            cursor = next_cursor

            facility_actions.append(
                FacilityActionFactory.create_facility_action(cut, facility))
        
        return AreaAction(
            facilities=facility_actions,
            consumed_ndarray_len=cursor
        )
