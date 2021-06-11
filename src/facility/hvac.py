from dataclasses import dataclass
import enum
from typing import Type
from xml.etree.ElementTree import Element
from src.environment import AreaEnvironment, ExternalEnvironment
from src.facility.facility_base import Facility, FacilityEffect, T


class HVACMode(enum.Enum):
    Off = enum.auto()
    Cool = enum.auto()
    Heat = enum.auto()


@dataclass
class HVAC(Facility):
    TYPE_STR = "HVAC"
    GUARD_TEMPERATURE = 3 # 3.2節のガード温度

    # static settings
    cool_max_power: float = 0 # [kW]
    heat_max_power: float = 0 # [kW]
    cool_cop: float = 0 # []
    heat_cop: float = 0 # []
    
    # internal state
    mode: HVACMode = HVACMode.Off

    # AI inputs
    status: bool = False
    temperature_setting: float = 0 # [℃]


    def change_setting(self, status: bool, temperature: float):
        self.status = status
        self.temperature_setting = temperature


    def update(self, area_temperature: float, **_) -> FacilityEffect:
        self.update_mode(area_temperature)

        temp_dif = area_temperature - self.temperature_setting

        clamp_to_01 = lambda x: min(max(x, 0), 1)

        if self.mode == HVACMode.Cool:
            power = clamp_to_01(temp_dif / HVAC.GUARD_TEMPERATURE) * self.cool_max_power
            heat = -self.cool_cop * power
        elif self.mode == HVACMode.Heat:
            power = clamp_to_01(-temp_dif / HVAC.GUARD_TEMPERATURE) * self.heat_max_power
            heat = self.heat_cop * power
        else:
            power = heat = 0
        
        return FacilityEffect(power=power, heat=heat)


    def update_mode(self, ext_temperature: float):
        """モードの更新
        """
        if self.status:
            abs_temp_dif = abs(self.temperature_setting - ext_temperature)

            if self.mode == HVACMode.Off or abs_temp_dif > HVAC.GUARD_TEMPERATURE:
                if self.temperature_setting > ext_temperature:
                    self.mode = HVACMode.Heat
                else:
                    self.mode = HVACMode.Cool
        else:
            self.mode = HVACMode.Off
    
    @classmethod
    def from_xml_element(cls: Type[T], elem: Element) -> T:
        facility = super(HVAC, cls).from_xml_element(elem)

        facility.cool_max_power = float(facility.params['cool-max-power'])
        facility.heat_max_power = float(facility.params['heat-max-power'])
        facility.cool_cop = float(facility.params['cool-cop'])
        facility.heat_cop = float(facility.params['heat-cop'])

        return facility

    def __repr__(self) -> str:
        return f"HVAC(mode={self.mode}, temp_setting={self.temperature_setting:.1f})"