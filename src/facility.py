from dataclasses import dataclass
import enum
from typing import NamedTuple, Type, TypeVar
from xml.etree.ElementTree import Element

from src.environment import AreaEnvironment, ExternalEnvironment


class FacilityType(enum.Enum):
    PV = enum.auto()
    HVAC = enum.auto()
    ES = enum.auto()


class FacilityEffect(NamedTuple):
    """設備によるエリアの働きかけ(2.2節)を表すオブジェクト
    """

    power: float
    heat: float


T = TypeVar('T', bound='Facility')

@dataclass
class Facility:
    """設備を表すオブジェクト
    """

    facility_type: FacilityType
    params: dict

    def update(self, ext_env: ExternalEnvironment, area_env: AreaEnvironment) -> FacilityEffect:
        """環境変数に応じて設備の状態を更新し、エリアへの影響を返す
        """

        effect_dict = {
            FacilityType.PV: FacilityEffect(
                power=-ext_env.solar_radiation/10, 
                heat=ext_env.solar_radiation),
            FacilityType.HVAC: FacilityEffect(
                power=area_env.people, 
                heat=-ext_env.temperature*(area_env.people+1)),
            FacilityType.ES: FacilityEffect(
                power=20-ext_env.electric_price_unit, 
                heat=10)
        }
        
        return effect_dict[self.facility_type]

    @classmethod
    def from_xml_element(cls: Type[T], elem: Element) -> T:
        assert elem.tag == "facility", f"invalid element for {cls}"

        child_params = filter(lambda child: child.tag == 'parameter', elem)

        return cls(
            facility_type=FacilityType[elem.attrib['type']],
            params=dict((param.attrib['name'], param.attrib['value']) for param in child_params)
        )
    