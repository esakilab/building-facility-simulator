from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Type, TypeVar
from xml.etree.ElementTree import Element

from simulator.environment import AreaEnvironment, ExternalEnvironment
from simulator.io import FacilityAction, FacilityState

class FacilityEffect(NamedTuple):
    """設備によるエリアの働きかけ(2.2節)を表すオブジェクト
    """

    power: float # [W/m^2]
    heat: float  # [kJ/s = kW]


T = TypeVar('T', bound='Facility')

@dataclass
class Facility(ABC):
    """設備を表す抽象クラス
    """
    
    params: dict[str, str]

    @abstractmethod
    def update(
        self, 
        action: FacilityAction,
        ext_env: ExternalEnvironment, 
        area_env: AreaEnvironment, 
        area_temperature: float,
    ) -> tuple[FacilityState, FacilityEffect]:

        """環境変数に応じて設備の状態を更新し、エリアへの影響を返す
        """
        
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_xml_element(cls: Type[T], elem: Element) -> T:
        assert elem.tag == "facility" and elem.attrib['type'] == cls.TYPE_STR, \
            f"invalid element for {cls}"

        child_params = filter(lambda child: child.tag == 'parameter', elem)

        return cls(
            params=dict((param.attrib['name'], param.attrib['value']) for param in child_params)
        )


    @property
    @classmethod
    @abstractmethod
    def TYPE_STR(cls) -> str:
        raise NotImplementedError()