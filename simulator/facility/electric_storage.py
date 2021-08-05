from dataclasses import dataclass
import enum
from typing import Type
from xml.etree.ElementTree import Element

from simulator.environment import AreaEnvironment, ExternalEnvironment
from simulator.facility.facility_base import Facility, FacilityEffect, FacilityState, T
from simulator.io import FacilityAction


class ESMode(enum.Enum):
    Standby = "stand_by"
    Charge = "charge"
    Discharge = "discharge"


@dataclass
class ESState(FacilityState):
    charge_ratio: float


@dataclass
class ElectricStorage(Facility):
    TYPE_STR = "ES"

    # static settings
    charge_power: float = 0 # [kW]
    discharge_power: float = 0 # [kW]
    capacity: float = 0 # [kWh]
    
    # internal state
    charge_ratio: float = 0

    # AI inputs
    mode: ESMode = ESMode.Standby


    def update_setting(self, action: FacilityAction):
        self.mode = ESMode(action.get("mode", self.mode.value))


    def update(self, action: FacilityAction, **_) -> tuple[ESState, FacilityEffect]:
        self.update_setting(action)

        if self.mode == ESMode.Charge and self.charge_ratio < 0.98:
            # status = charge
            self.charge_ratio += (self.charge_power / 60) / self.capacity
            power = self.charge_power

        elif self.mode == ESMode.Discharge and self.charge_ratio > 0.03:
            # status = discharge
            self.charge_ratio -= (self.charge_power / 60) / self.capacity
            power = -self.discharge_power
        
        else:
            # status = no-operation
            power = 0
        
        self.charge_ratio = min(max(self.charge_ratio, 0), 1)

        return (
            ESState(charge_ratio=self.charge_ratio), 
            FacilityEffect(power=power, heat=0))

    
    @classmethod
    def from_xml_element(cls: Type[T], elem: Element) -> T:
        facility = super(ElectricStorage, cls).from_xml_element(elem)

        facility.charge_power = float(facility.params['charge-power'])
        facility.discharge_power = float(facility.params['discharge-power'])
        facility.capacity = float(facility.params['capacity'])

        return facility


    def __repr__(self) -> str:
        return f"ES(charge_ratio={self.charge_ratio:.3f}, mode={self.mode})"
