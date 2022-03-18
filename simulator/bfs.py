from __future__ import annotations
from copy import deepcopy
from datetime import timedelta, datetime
from typing import Callable, Type, TypeVar

import numpy as np

from simulator.area import Area
from simulator.building import BuildingAction, BuildingState
from simulator.environment import AreaEnvironment, ExternalEnvironment
from simulator.interfaces.config import AreaAttributes, SimulatorConfig
from simulator.interfaces.model import RlModel


class BuildingFacilitySimulator:
    """シミュレータを表すオブジェクトで、外部プログラムとのやり取りを担当
    """

    # TODO: モデルが複数になると報酬が複数になりそう
    def __init__(self, config: SimulatorConfig, calc_reward: Callable[[BuildingState, BuildingAction], np.ndarray]):
        self.areas: list[Area] = list(map(AreaAttributes.to_area, config.building_attributes.areas))
        self.ext_envs: list[ExternalEnvironment] = config.external_enviroment_time_series
        self.area_envs: list[list[AreaEnvironment]] = \
            list(map(lambda area: area.area_environment_time_series, config.building_attributes.areas))

        self.calc_reward: Callable[[BuildingState, BuildingAction], float] = calc_reward
        
        self.start_time: datetime = config.start_time
        self.cur_steps: int = 0
        self.total_steps: int = min(0, len(self.ext_envs), *filter(None, map(len, self.area_envs)))


    M = TypeVar('M', bound=RlModel)

    def create_rl_model(self, ModelClass: Type[M], **kwargs) -> M:
        return ModelClass(
            state_shape=self.get_state_shape(),
            action_shape=self.get_action_shape(),
            **kwargs
        )


    def get_area_env(self, area_id: int, timestamp: int):
        if area_id in self.area_envs:
            return self.area_envs[area_id][timestamp]
        else:
            return AreaEnvironment.empty()

    
    def has_finished(self):
        return self.cur_steps == self.total_steps

        
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

        if self.has_finished():
            return (None, None)

        ext_env = self.ext_envs[self.cur_steps]

        for area_id, area in enumerate(self.areas):
            area.update(action.areas[area_id], ext_env, self.get_area_env(area_id, self.cur_steps))

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

        if train_model:
            model.add_to_buffer(state, action, next_state, reward)

        return next_state, action, reward

    
    def get_current_datetime(self):
        return self.start_time + timedelta(minutes=self.cur_steps)


    def get_state(self) -> BuildingState:
        area_states = [area.get_state() for area in self.areas]
        if self.cur_steps < len(self.ext_envs):
            ext_env = self.ext_envs[self.cur_steps]
        else:
            ext_env = self.ext_envs[-1]
            
        return BuildingState.create(area_states, ext_env)


    def get_state_shape(self) -> tuple[int]:
        return self.get_state().to_ndarray().shape

    
    def get_action_shape(self) -> tuple[int]:
        size = 0
        for area in self.areas:
            for facility in area.facilities:
                size += facility.ACTION_TYPE.NDARRAY_SHAPE[0]
        
        return (size,)


    def print_cur_state(self):
        print(f"\niteration {self.cur_steps} ({self.get_current_datetime()})")
        if not self.has_finished():
            print(self.ext_envs[self.cur_steps])

        for aid, (area, st) in enumerate(zip(self.areas, self.last_state.areas)):
            print(f"area {aid}: temp={area.temperature:.2f}, power={st.power_consumption:.2f}, {area.facilities[0]}")

        print(f"total power consumption: {self.last_state.power_balance:.2f}", flush=True)
    

    def __add__(self, other: BuildingFacilitySimulator) -> BuildingFacilitySimulator:
        assert self.area_envs.keys() == other.area_envs.keys()
        result = deepcopy(self)

        result.total_steps += other.total_steps
        result.ext_envs += other.ext_envs
        for key in result.area_envs:
            result.area_envs[key] += other.area_envs[key]

        return result
    

    def __mul__(self, other: int) -> BuildingFacilitySimulator:
        result = deepcopy(self)

        result.total_steps *= other
        result.ext_envs *= other
        for key in result.area_envs:
            result.area_envs[key] *= other
        
        return result
