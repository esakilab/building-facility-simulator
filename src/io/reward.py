from typing import NamedTuple, Type, TypeVar
import numpy as np

from src.io.state import AreaState, BuildingState

T = TypeVar('T', bound='Reward')

class Reward(NamedTuple):
    """報酬を表すタプルオブジェクト
    """

    metric1: float


    @classmethod
    def calc_metrix1(cls, state: BuildingState) -> float:
        LAMBDA1 = 0.2
        LAMBDA2 = 0.1
        LAMBDA3 = 0.1
        LAMBDA4 = 20
        T_MAX = 30
        T_MIN = 20
        T_TARGET = 25

        area_temp = np.array([area.temperature for area in state.areas])
        
        reward = np.exp(-LAMBDA1 * (area_temp - T_TARGET) ** 2).sum()
        reward += - LAMBDA2 * (np.where((T_MIN - area_temp) < 0, 0, (T_MIN - area_temp)).sum())
        reward += - LAMBDA2 * (np.where((area_temp - T_MAX) < 0, 0, (area_temp - T_MAX)).sum())
        reward += - LAMBDA3 * state.electric_price_unit
        # reward += LAMBDA4 * state.charge_ratio

        '''
        print(np.exp(-lambda1 * (temp - T_target) ** 2).sum())
        print(-lambda2 * (np.where((T_min - temp) < 0, 0, (T_min - temp)).sum()))
        print(-lambda2 * (np.where((temp - T_max) < 0, 0, (temp - T_max)).sum()))
        print(-lambda3 * electric_price_unit)
        print(lambda4*charge_ratio)
        '''

        # 人数の項を考える
        # 太陽光の発電状況
        # 蓄電池の残量
        # 異なるrewardを考えた状況設定

        return reward


    @classmethod
    def from_state(cls: Type[T], state: BuildingState) -> T:
        return cls(
            metric1=cls.calc_metrix1(state)
        )
    