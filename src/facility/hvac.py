from dataclasses import dataclass
import enum
from typing import NamedTuple, Type
from xml.etree.ElementTree import Element

from src.environment import ExternalEnvironment
from src.facility.facility_base import Facility, FacilityEffect, T
from src.io.action import FacilityAction


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
class HVACState:
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

                self.stand_by = (calc_deficit() < HVACState.STAND_BY_GUARD_TEMPERATURE)

            else:
                # 状態遷移
                if self.mode == HVACMode.Heat:
                    deficit = set_temperature - area_temperature
                else:
                    deficit = area_temperature - set_temperature
                

                if deficit < -HVACState.MODE_GUARD_TEMPERATURE:
                    self.mode = self.mode.switch()
                
                if self.stand_by and deficit >= HVACState.STAND_BY_GUARD_TEMPERATURE:
                    self.stand_by = False
                elif not self.stand_by and deficit <= -HVACState.STAND_BY_GUARD_TEMPERATURE:
                    self.stand_by = True
            
        else:
            self.mode = HVACMode.Off
    
    def is_running(self):
        return (self.mode != HVACMode.Off and not self.stand_by)
        

@dataclass
class HVAC(Facility):
    TYPE_STR = "HVAC"
    EFFICIENCY_THRESH_TEMPERATURE = 10

    # static settings
    cool_max_power: float = 0 # [kW]
    heat_max_power: float = 0 # [kW]
    cool_cop: float = 0 # []
    heat_cop: float = 0 # []
    
    # internal state
    state: HVACState = HVACState()

    # AI inputs
    status: bool = False
    set_temperature: float = 0 # [℃]


    def update_setting(self, action: FacilityAction):
        self.status = bool(action.get("status", self.status))
        self.set_temperature = float(action.get("temperature", self.set_temperature))


    def update(self, action: FacilityAction, ext_env: ExternalEnvironment, area_temperature: float, **_) -> FacilityEffect:
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
            
            return FacilityEffect(
                power=power, 
                heat=heat_coef * cop * power
            )
        
        else:
            return FacilityEffect(power=0, heat=0)

    
    @classmethod
    def from_xml_element(cls: Type[T], elem: Element) -> T:
        facility = super(HVAC, cls).from_xml_element(elem)

        facility.cool_max_power = float(facility.params['cool-max-power'])
        facility.heat_max_power = float(facility.params['heat-max-power'])
        facility.cool_cop = float(facility.params['cool-cop'])
        facility.heat_cop = float(facility.params['heat-cop'])
        facility.state = HVACState()

        return facility

    def __repr__(self) -> str:
        return f"HVAC(mode={self.state.mode}, stand_by={self.state.stand_by}, temp_setting={self.set_temperature:.1f})"