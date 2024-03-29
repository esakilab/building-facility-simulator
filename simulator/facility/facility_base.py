from __future__ import annotations
from abc import ABC, abstractclassmethod, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Type, TypeVar

import numpy as np
from pydantic import BaseModel

from simulator.environment import AreaEnvironment, ExternalEnvironment


class FacilityEffect(NamedTuple):
    """設備によるエリアの働きかけ(2.2節)を表すオブジェクト
    """

    power: float # [kW]
    heat: float  # [kJ/s = kW]


T = TypeVar('T', bound='Facility')

class Facility(ABC, BaseModel):
    """設備を表す抽象クラス
    """

    @abstractmethod
    def update(
        self, 
        action: FacilityAction,
        ext_env: ExternalEnvironment,
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


    @property
    @classmethod
    @abstractmethod
    def STATE_TYPE(cls) -> type[FacilityState]:
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


    @property
    @abstractclassmethod
    def NDARRAY_SHAPE(cls) -> tuple[int]:
        raise NotImplementedError()


class EmptyFacilityState(FacilityState):
    NDARRAY_SHAPE = (0,)
    
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
