from __future__ import annotations
from dataclasses import dataclass
import enum
from typing import Type

import numpy as np

from simulator.facility.facility_base import Facility, FacilityAction, FacilityEffect, FacilityState, T
from simulator.facility.factory import FacilityFactory


class ESMode(enum.Enum):
    Standby = "stand_by"
    Charge = "charge"
    Discharge = "discharge"

    def from_int(src: int) -> ESMode:
        if src > 1 / 3:
            return ESMode.Charge
        elif src > -1 / 3:
            return ESMode.Standby
        else:
            return ESMode.Discharge


@dataclass
class ESState(FacilityState):
    NDARRAY_SHAPE = (1,)
    
    charge_ratio: float


    def to_ndarray(self) -> np.ndarray:
        return np.array([self.charge_ratio])


@dataclass
class ESAction(FacilityAction):
    NDARRAY_SHAPE = (1,)

    mode: ESMode

    @classmethod
    def from_ndarray(cls, src: np.ndarray) -> ESAction:
        return cls(mode=ESMode.from_int(int(src[0])))


@FacilityFactory.register("ES")
class ElectricStorage(Facility):
    STATE_TYPE = ESState
    ACTION_TYPE = ESAction

    # static settings
    charge_power: float # [kW]
    discharge_power: float # [kW]
    capacity: float # [kWh]
    
    # internal state
    charge_ratio: float = 0

    # AI inputs
    mode: ESMode = ESMode.Standby


    def update_setting(self, action: ESAction):
        self.mode = action.mode


    def update(self, action: ESAction, **_) -> tuple[ESState, FacilityEffect]:
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
            self.get_state(), 
            FacilityEffect(power=power, heat=0))


    def get_state(self) -> FacilityState:
        return ESState(charge_ratio=self.charge_ratio)


    def __str__(self) -> str:
        return f"ES(charge_ratio={self.charge_ratio:.3f}, mode={self.mode})"
