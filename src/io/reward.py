from typing import NamedTuple, Type, TypeVar
import numpy as np

from src.io.state import AreaState, BuildingState

T = TypeVar('T', bound='Reward')

class Reward(NamedTuple):
    """報酬を表すタプルオブジェクト
    """
    LAMBDA1 = 0.2
    LAMBDA2 = 0.1
    LAMBDA3 = 0.1
    T_MAX = 30
    T_MIN = 20
    T_TARGET = 25

    metric1: float


    @classmethod
    def calc_metrix1(cls, state: BuildingState) -> float:
        area_temp = np.array([area.temperature for area in state.areas])
        
        reward = np.exp(-cls.LAMBDA1 * (area_temp - cls.T_TARGET) ** 2).sum()
        reward += - cls.LAMBDA2 * (np.where((cls.T_MIN - area_temp) < 0, 0, (cls.T_MIN - area_temp)).sum())
        reward += - cls.LAMBDA2 * (np.where((area_temp - cls.T_MAX) < 0, 0, (area_temp - cls.T_MAX)).sum())
        reward += - cls.LAMBDA3 * state.electric_price_unit

        return reward


    @classmethod
    def from_state(cls: Type[T], state: BuildingState) -> T:
        return cls(
            metric1=cls.calc_metrix1(state)
        )
    