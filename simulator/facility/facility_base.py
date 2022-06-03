from __future__ import annotations
from abc import ABC, abstractclassmethod, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Type, TypeVar
from xml.etree.ElementTree import Element

import numpy as np

from simulator.environment import AreaEnvironment, ExternalEnvironment


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

    @abstractmethod
    def get_state(self) -> FacilityState:

        """現在の施設の状態を返す
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


    @property
    @classmethod
    @abstractmethod
    def ACTION_TYPE(cls) -> type[FacilityAction]:
        raise NotImplementedError()


class FacilityState(ABC):
    """AIに出力する、設備の状態を表す抽象クラス
    """
    @abstractmethod
    def to_ndarray(self) -> np.ndarray:
        raise NotImplementedError()


class EmptyFacilityState(FacilityState):
    def to_ndarray(self) -> np.ndarray:
        return np.array([])


class FacilityAction(ABC):

    @abstractclassmethod
    def from_ndarray(cls, src: np.ndarray) -> FacilityAction:
        raise NotImplementedError()


    @property
    @abstractclassmethod
    def NDARRAY_SHAPE(cls) -> tuple[int]:
        raise NotImplementedError()


@dataclass
class EmptyFacilityAction(FacilityAction):
    NDARRAY_SHAPE = (0,)

    @classmethod
    def from_ndarray(cls, src: np.ndarray) -> EmptyFacilityAction:
        return cls()


class FacilityActionFactory:
    @staticmethod
    def create_facility_action(src: np.ndarray, facility: Facility) -> FacilityAction:
        assert facility.ACTION_TYPE.NDARRAY_SHAPE == src.shape, \
            f"Shape mismatch on initializing {facility.ACTION_TYPE}. ({facility.ACTION_TYPE.NDARRAY_SHAPE} != {src.shape})"

        return facility.ACTION_TYPE.from_ndarray(src=src)
