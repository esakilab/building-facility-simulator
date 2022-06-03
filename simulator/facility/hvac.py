from __future__ import annotations
from dataclasses import dataclass
import enum
from typing import ClassVar

import numpy as np

from simulator.environment import ExternalEnvironment
from simulator.facility.facility_base import EmptyFacilityState, Facility, FacilityAction, FacilityEffect, T, FacilityState
from simulator.facility.factory import FacilityFactory


class HVACMode(enum.Enum):
    Off = enum.auto()
    Cool = enum.auto()
    Heat = enum.auto()

    def switch(self):
        return {
            HVACMode.Cool: HVACMode.Heat,
            HVACMode.Heat: HVACMode.Cool,
            HVACMode.Off: HVACMode.Off
        }[self]

        

@dataclass
class HVACStateInternal:
    """HVACの内部状態を表すオブジェクト
    3.2節の内部状態の遷移が複雑なため、HVACクラスから分離した
    """
    # 3.2節のガード温度
    MODE_GUARD_TEMPERATURE = 3 
    STAND_BY_GUARD_TEMPERATURE = 0.5


    mode: HVACMode = HVACMode.Off # opmode
    stand_by: bool = False # !onpump


    def update(self, is_on: bool, area_temperature: float, set_temperature: float):
        """内部状態の更新
        """
        def calc_deficit():
            if self.mode == HVACMode.Heat:
                return set_temperature - area_temperature
            else:
                return area_temperature - set_temperature
        
        if is_on:
            abs_temp_dif = abs(set_temperature - area_temperature)

            if self.mode == HVACMode.Off:
                # 初期値
                if set_temperature > area_temperature:
                    self.mode = HVACMode.Heat
                else:
                    self.mode = HVACMode.Cool

                self.stand_by = (calc_deficit() < HVACStateInternal.STAND_BY_GUARD_TEMPERATURE)

            else:
                # 状態遷移
                if self.mode == HVACMode.Heat:
                    deficit = set_temperature - area_temperature
                else:
                    deficit = area_temperature - set_temperature
                

                if deficit < -HVACStateInternal.MODE_GUARD_TEMPERATURE:
                    self.mode = self.mode.switch()
                
                if self.stand_by and deficit >= HVACStateInternal.STAND_BY_GUARD_TEMPERATURE:
                    self.stand_by = False
                elif not self.stand_by and deficit <= -HVACStateInternal.STAND_BY_GUARD_TEMPERATURE:
                    self.stand_by = True
            
        else:
            self.mode = HVACMode.Off
    
    def is_running(self):
        return (self.mode != HVACMode.Off and not self.stand_by)


@dataclass
class HVACAction(FacilityAction):
    NDARRAY_SHAPE = (2,)

    status: bool
    set_temperature: float
    
    @classmethod
    def from_ndarray(cls, src: np.ndarray) -> HVACAction:
        return cls(
            status=(src[0] > 0.), 
            set_temperature=int(src[1] * 7.5 + 22.5)
        )


@FacilityFactory.register("HVAC")
class HVAC(Facility):
    STATE_TYPE = EmptyFacilityState
    ACTION_TYPE = HVACAction

    EFFICIENCY_THRESH_TEMPERATURE: ClassVar[int] = 10

    # static settings
    cool_max_power: float # [kW]
    heat_max_power: float # [kW]
    cool_cop: float # []
    heat_cop: float # []
    
    # internal state
    state: HVACStateInternal = HVACStateInternal()

    # AI inputs
    status: bool = False
    set_temperature: float = 0 # [℃]


    def update_setting(self, action: HVACAction):
        self.status = action.status
        self.set_temperature = action.set_temperature


    def update(self, action: HVACAction, ext_env: ExternalEnvironment, 
            area_temperature: float, **_) -> tuple[FacilityState, FacilityEffect]:

        self.update_setting(action)

        self.state.update(self.status, area_temperature, self.set_temperature)

        if self.state.is_running():

            mode_to_vars = {
                HVACMode.Cool: (ext_env.temperature - area_temperature, -self.cool_cop, self.cool_max_power),
                HVACMode.Heat: (area_temperature - ext_env.temperature, self.heat_cop, self.heat_max_power),
            }

            outside_deficit, cop, power = mode_to_vars[self.state.mode]

            if outside_deficit >= HVAC.EFFICIENCY_THRESH_TEMPERATURE:
                heat_coef = 1
            
            else:
                heat_coef = 2 - max(0, outside_deficit / HVAC.EFFICIENCY_THRESH_TEMPERATURE)
            
            # print(f"heat: {heat_coef * cop * power}, state: {self.state}")
            
            effect = FacilityEffect(
                power=power, 
                heat=heat_coef * cop * power
            )
        
        else:
            effect = FacilityEffect(power=0, heat=0)
        
        return (self.get_state(), effect)


    def get_state(self) -> EmptyFacilityState:
        return EmptyFacilityState()

    def __str__(self) -> str:
        return f"HVAC(mode={self.state.mode}, stand_by={self.state.stand_by}, temp_setting={self.set_temperature:.1f})"