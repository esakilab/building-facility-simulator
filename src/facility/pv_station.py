from dataclasses import dataclass
from typing import Type
from xml.etree.ElementTree import Element
from src.environment import AreaEnvironment, ExternalEnvironment
from src.facility.facility_base import Facility, FacilityEffect, T

@dataclass
class PVStation(Facility):
    TYPE_STR = "PV"

    # static settings
    max_power: float = 0 # [kW]

    def change_setting(self):
        pass

    def update(self, ext_env: ExternalEnvironment, **_) -> FacilityEffect:
        return FacilityEffect(
            power = -self.max_power * ext_env.solar_radiation / 1000,
            heat  = 0
        )
    
    @classmethod
    def from_xml_element(cls: Type[T], elem: Element) -> T:
        facility = super(PVStation, cls).from_xml_element(elem)
        facility.max_power = float(facility.params['max-power'])

        return facility
    

    def __repr__(self) -> str:
        return f"PV"