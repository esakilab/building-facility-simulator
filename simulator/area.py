from dataclasses import dataclass
from typing import Type, TypeVar
from xml.etree.ElementTree import Element

from simulator.facility import Facility, xml_element_to_facility
from simulator.environment import ExternalEnvironment, AreaEnvironment
from simulator.io import AreaAction, AreaState


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
    

    def update(
        self, 
        action: AreaAction,
        ext_env: ExternalEnvironment, 
        area_env: AreaEnvironment = AreaEnvironment.empty(),
    ) -> AreaState:
        """ext_envとarea_envに応じて温度と消費電力を更新する

        2.6節の2,3に対応
        """

        beta = area_env.calc_beta()
        power_consumption = 0.

        facility_states = []

        for fid, facility in enumerate(self.facilities):
            state, effect = facility.update(
                action=action[fid],
                ext_env=ext_env, 
                area_env=area_env, 
                area_temperature=self.temperature)

            facility_states.append(state)

            beta += effect.heat * 60
            power_consumption += effect.power

        if self.simulate_temperature:
            temp_dif = self.temperature - ext_env.temperature
            self.temperature += -ALPHA * temp_dif + beta / (self.capacity * 1.189)

        else:
            self.temperature = ext_env.temperature

        return AreaState(
            power_consumption=power_consumption,
            temperature=self.temperature,
            people=area_env.people,
            facilities=facility_states
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