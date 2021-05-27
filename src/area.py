from dataclasses import dataclass
from typing import List, Type, TypeVar
from xml.etree.ElementTree import Element
from src.facility import Facility
from src.environment import ExternalEnvironment, AreaEnvironment


ALPHA = 0.3

T = TypeVar('T', bound='Area')

@dataclass
class Area:
    """エリアを表すオブジェクト
    """

    name: str
    facilities: List[Facility]
    capacity: float
    temperature: float
    power_consumption: float = 0
    
    def update(self, ext_env: ExternalEnvironment, area_env: AreaEnvironment):
        """ext_envとarea_envに応じて温度と消費電力を更新する

        2.6節の2,3に対応
        """

        beta = area_env.calc_beta()
        self.power_consumption = 0.

        for facility in self.facilities:
            effect = facility.update(ext_env, area_env)

            beta += effect.heat
            self.power_consumption += effect.power

        temp_dif = self.temperature - ext_env.temperature
        self.temperature += (-ALPHA * temp_dif + beta) / self.capacity
    

    @classmethod
    def from_xml_element(cls: Type[T], elem: Element) -> T:
        assert elem.tag == "area", f"invalid element for {cls}"

        child_facilities = filter(lambda child: child.tag == 'facility', elem)

        return cls(
            name=elem.attrib['name'],
            facilities=list(map(Facility.from_xml_element, child_facilities)),
            capacity=float(elem.attrib['capacity']),
            temperature=float(elem.attrib.get('temperature', "25"))
        )