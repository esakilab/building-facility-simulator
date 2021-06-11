from dataclasses import dataclass
from typing import Type, TypeVar
from xml.etree.ElementTree import Element
from src.facility import Facility, xml_element_to_facility
from src.environment import ExternalEnvironment, AreaEnvironment


ALPHA = 0.3

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
    
    def update(self, ext_env: ExternalEnvironment, 
            area_env: AreaEnvironment = AreaEnvironment.empty()):
        """ext_envとarea_envに応じて温度と消費電力を更新する

        2.6節の2,3に対応
        """

        beta = area_env.calc_beta()
        self.power_consumption = 0.

        for facility in self.facilities:
            effect = facility.update(
                ext_env=ext_env, 
                area_env=area_env, 
                area_temperature=self.temperature)

            beta += effect.heat
            self.power_consumption += effect.power


        if self.simulate_temperature:
            temp_dif = self.temperature - ext_env.temperature
            self.temperature += (-ALPHA * temp_dif + beta) / (self.capacity * 1.189)

        else:
            self.temperature = ext_env.temperature
    

    @classmethod
    def from_xml_element(cls: Type[T], elem: Element) -> T:
        assert elem.tag == "area", f"invalid element for {cls}"

        child_facilities = filter(lambda child: child.tag == 'facility', elem)

        return cls(
            name=elem.attrib['name'],
            facilities=list(map(xml_element_to_facility, child_facilities)),
            simulate_temperature=('capacity' in elem.attrib),
            capacity=float(elem.attrib.get('capacity', "-1")),
            temperature=float(elem.attrib.get('temperature', "25"))
        )