from dataclasses import dataclass
from typing import Type
from xml.etree.ElementTree import Element

from simulator.environment import AreaEnvironment, ExternalEnvironment
from simulator.facility.facility_base import Facility, FacilityEffect, T
from simulator.io import FacilityState

@dataclass
class PVStation(Facility):
    TYPE_STR = "PV"

    # static settings
    max_power: float = 0 # [kW]

    def update(self, ext_env: ExternalEnvironment, **_) -> tuple[FacilityState, FacilityEffect]:
        effect = FacilityEffect(
            power = -self.max_power * ext_env.solar_radiation / 1000,
            heat  = 0
        )
        return (self.get_state(), effect)


    def get_state(self) -> FacilityState:
        return FacilityState.empty()
    

    @classmethod
    def from_xml_element(cls: Type[T], elem: Element) -> T:
        facility = super(PVStation, cls).from_xml_element(elem)
        facility.max_power = float(facility.params['max-power'])

        return facility
    

    def __repr__(self) -> str:
        return f"PV"