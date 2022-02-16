from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple, Type, TypeVar
from xml.etree.ElementTree import Element

import numpy as np

from simulator.facility import Facility, xml_element_to_facility
from simulator.environment import ExternalEnvironment, AreaEnvironment
from simulator.facility.facility_base import FacilityActionFactory, FacilityState


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
        area_env: AreaEnvironment = AreaEnvironment.empty(),
    ) -> AreaState:
        """ext_envとarea_envに応じて温度と消費電力を更新する

        2.6節の2,3に対応
        """
        self.people = area_env.people

        beta = area_env.calc_beta()
        self.power_consumption = 0.

        for fid, facility in enumerate(self.facilities):
            state, effect = facility.update(
                action=action.facilities[fid],
                ext_env=ext_env, 
                area_env=area_env, 
                area_temperature=self.temperature)

            beta += effect.heat * 60
            self.power_consumption += effect.power

        if self.simulate_temperature:
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


    @classmethod
    def from_xml_element(cls: Type[T], elem: Element) -> T:
        assert elem.tag == "area", f"invalid element for {cls}"

        facility_elems = filter(lambda child: child.tag == 'facility', elem)

        facilities = []

        for facility_elem in sorted(facility_elems, key=lambda elem: elem.attrib['id']):
            assert int(facility_elem.attrib['id']) == len(facilities), \
                "Area IDs must start from 0 and must be consecutive."

            facilities.append(xml_element_to_facility(facility_elem))

        return cls(
            name=elem.attrib['name'],
            facilities=facilities,
            simulate_temperature=('capacity' in elem.attrib),
            capacity=float(elem.attrib.get('capacity', "-1")),
            temperature=float(elem.attrib.get('temperature', "25"))
        )


class AreaState(NamedTuple):
    """エリアの状態を表すオブジェクト
    """

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
class AreaAction():
    facilities: list[FacilityActionFactory]

    def from_ndarray(src: np.ndarray, facilities: list[Facility]) -> AreaAction:
        facility_actions = []
        for facility in facilities:
            cut = src[:facility.ACTION_TYPE.NDARRAY_SHAPE[0]]
            src = src[facility.ACTION_TYPE.NDARRAY_SHAPE[0]:]

            facility_actions.append(
                FacilityActionFactory.create_facility_action(cut, facility))
        
        return AreaAction(facilities=facility_actions)
