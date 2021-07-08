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
    def from_state(cls: Type[T], state: BuildingState) -> T:
        metric1 = []
        
        for area in state.areas:
            temp_reward1 = np.exp(-cls.LAMBDA1 * (area.temperature - cls.T_TARGET)**2)
            temp_reward2 = \
                -cls.LAMBDA2 * (max(0, cls.T_MIN - area.temperature) + max(0, area.temperature - cls.T_MAX))

            electric_reward = cls.LAMBDA3 * area.power_consumption * state.electric_price_unit

            metric1.append(temp_reward1 + temp_reward2 + electric_reward)

        return cls(
            metric1=(metric1)
        )