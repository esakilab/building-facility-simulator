from dataclasses import dataclass
import enum
from typing import Type
from xml.etree.ElementTree import Element
from src.environment import AreaEnvironment, ExternalEnvironment
from src.facility.facility_base import Facility, FacilityEffect, T


class ESMode(enum.Enum):
    Standby = enum.auto()
    Charge = enum.auto()
    Discharge = enum.auto()


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


    def start_charging(self):
        self.mode = ESMode.Charge

    def start_discharging(self):
        self.mode = ESMode.Discharge

    def start_standing_by(self):
        self.mode = ESMode.Standby


    def update(self, **_) -> FacilityEffect:
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

        return FacilityEffect(power=power, heat=0)

    
    @classmethod
    def from_xml_element(cls: Type[T], elem: Element) -> T:
        facility = super(ElectricStorage, cls).from_xml_element(elem)

        facility.charge_power = float(facility.params['charge-power'])
        facility.discharge_power = float(facility.params['discharge-power'])
        facility.capacity = float(facility.params['capacity'])

        return facility


    def __repr__(self) -> str:
        return f"ES(charge_ratio={self.charge_ratio:.3f})"