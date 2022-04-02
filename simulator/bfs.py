from __future__ import annotations
from datetime import timedelta, datetime
from itertools import chain, count
from typing import Callable, Iterator, Optional, Type, TypeVar

import numpy as np

from simulator.area import Area
from simulator.building import BuildingAction, BuildingState
from simulator.environment import AreaEnvironment, BuildingEnvironment, ExternalEnvironment
from simulator.interfaces.config import AreaAttributes, SimulatorConfig
from simulator.interfaces.model import RlModel


class BuildingFacilitySimulator:
    """シミュレータを表すオブジェクトで、外部プログラムとのやり取りを担当
    """

    # TODO: モデルが複数になると報酬が複数になりそう
    def __init__(self, config: SimulatorConfig, calc_reward: Callable[[BuildingState, BuildingAction], np.ndarray]):
        # NOTE: 実験では太陽光パネルとエアコンの部屋一つだけ使用する
        config.building_attributes.areas = config.building_attributes.areas[:2]

        self.areas: list[Area] = list(map(AreaAttributes.to_area, config.building_attributes.areas))

        self.env_iter: Iterator[BuildingEnvironment] = config.get_env_iter()
        self.prev_env: Optional[BuildingEnvironment] = None
        self.next_env: Optional[BuildingEnvironment] = next(self.env_iter, None)

        self.calc_reward: Callable[[BuildingState, BuildingAction], float] = calc_reward
        
        self.start_time: datetime = config.start_time
        self.cur_steps: int = 0


    M = TypeVar('M', bound=RlModel)

    def create_rl_model(self, ModelClass: Type[M], **kwargs) -> M:
        return ModelClass(
            state_shape=self.get_state_shape(),
            action_shape=self.get_action_shape(),
            **kwargs
        )


    def _next_step(self) -> Optional[BuildingEnvironment]:
        if self.next_env:
            self.prev_env = self.next_env
            self.next_env = next(self.env_iter, None)
            return self.prev_env
        else:
            return None


    def has_finished(self) -> bool:
        return not bool(self.next_env)


    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """2.6節のシミュレーションを1サイクル分進めるメソッド
        while not bfs.has_finished():
            for i in range(10):
                action = compute_action()
                (state, reward) = bfs.step(action)
            
            update_model()

        みたいにすると、10stepごとにモデルの更新を行える
        """

        action: BuildingAction = BuildingAction.from_ndarray(action, self.areas)

        if (cur_env := self._next_step()) is None:
            return (None, None)

        for area_id, area, area_env in zip(count(), self.areas, cur_env.areas):
            area.update(action.areas[area_id], cur_env.external, area_env)

        state = self.get_state()

        self.cur_steps += 1
        self.last_state = state

        return (
            state.to_ndarray(),
            self.calc_reward(state, action)
        )

    
    def step_with_model(self, model: RlModel, train_model: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = self.get_state().to_ndarray()
        action = model.select_action(state)
        next_state, reward = self.step(action)

        if next_state is not None and reward is not None and train_model:
            model.add_to_buffer(state, action, next_state, reward)

        return next_state, action, reward

    
    def get_current_datetime(self):
        return self.start_time + timedelta(minutes=self.cur_steps)


    def get_state(self) -> BuildingState:
        area_states = [area.get_state() for area in self.areas]

        cur_env = self.next_env or self.prev_env

        return BuildingState.create(area_states, cur_env.external)


    def get_state_shape(self) -> tuple[int]:
        return (BuildingState.NDARRAY_ELEMS + sum(a.get_state_shape()[0] for a in self.areas),)

    
    def get_action_shape(self) -> tuple[int]:
        size = 0
        for area in self.areas:
            for facility in area.facilities:
                size += facility.ACTION_TYPE.NDARRAY_SHAPE[0]
        
        return (size,)


    def print_cur_state(self):
        print(f"\niteration {self.cur_steps} ({self.get_current_datetime()})")
        if self.prev_env:
            print(self.prev_env.external)

        for aid, (area, st) in enumerate(zip(self.areas, self.last_state.areas)):
            print(f"area {aid}: temp={area.temperature:.2f}, power={st.power_consumption:.2f}, {area.facilities[0]}")

        print(f"total power consumption: {self.last_state.power_balance:.2f}", flush=True)
    

    # TODO: __init__とのコードのダブりをどうにかする
    @staticmethod
    def _from_models(
        areas: list[Area], 
        envs: list[BuildingEnvironment], 
        calc_reward: Callable[[BuildingState, BuildingAction], np.ndarray],
        start_time: datetime
    ) -> BuildingFacilitySimulator:
        bfs = BuildingFacilitySimulator.__new__(BuildingFacilitySimulator)
        bfs.areas = areas

        bfs.env_iter = iter(envs)
        bfs.prev_env = None
        bfs.next_env = next(bfs.env_iter, None)

        bfs.calc_reward = calc_reward

        bfs.start_time = start_time
        bfs.cur_steps = 0

        return bfs
