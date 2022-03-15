from typing import Type

from simulator.environment import ExternalEnvironment
from simulator.facility.facility_base import EmptyFacilityAction, EmptyFacilityState, Facility, FacilityEffect, T, FacilityState
from simulator.facility.factory import FacilityFactory


@FacilityFactory.register("PV")
class PVStation(Facility):
    ACTION_TYPE = EmptyFacilityAction

    # static settings
    max_power: float # [kW]

    def update(self, ext_env: ExternalEnvironment, **_) -> tuple[FacilityState, FacilityEffect]:
        effect = FacilityEffect(
            power = -self.max_power * ext_env.solar_radiation / 1000,
            heat  = 0
        )
        return (self.get_state(), effect)


    def get_state(self) -> EmptyFacilityState:
        return EmptyFacilityState()
    

    def __str__(self) -> str:
        return f"PV"