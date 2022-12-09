from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from itertools import islice
from typing import Callable, Iterator, NamedTuple, Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from simulator.area import Area, AreaState
from simulator.bfs import BuildingFacilitySimulator
from simulator.building import BuildingAction, BuildingState
from simulator.environment import BuildingEnvironment
from simulator.interfaces.config import SimulatorConfig
from simulator.interfaces.model import RlModel


class RemoteSimulatonManager:
    def __init__(
            self, 
            config: SimulatorConfig, 
            calc_reward: Callable[[BuildingState, BuildingAction], np.ndarray],
            summary_dir: Optional[str]):
        bfs = BuildingFacilitySimulator(config=config, calc_reward=calc_reward)

        self.env_iter: Iterator[BuildingEnvironment] = bfs.env_iter
        self.areas: list[Area] = bfs.areas
        self.start_dt: datetime = bfs.start_time
        self.current_dt: datetime = bfs.start_time
        self.calc_reward: Callable[[BuildingState, BuildingAction], np.ndarray] = bfs.calc_reward
        self.summary_writer: Optional[SummaryWriter] = SummaryWriter(summary_dir) if summary_dir else None

    def create_agent(self, model: RlModel, train_start_dt: datetime, end_dt: datetime):
        total_steps = int((end_dt - self.current_dt).total_seconds()) // 60

        bfs = BuildingFacilitySimulator._from_models(
            areas=self.areas,
            envs=list(islice(self.env_iter, total_steps)),
            calc_reward=self.calc_reward,
            start_time=self.current_dt
        )

        return RemoteSimulaionAgent(bfs=bfs, model=model, train_start_dt=train_start_dt)

    def load_checkpoint(self, checkpoint: RemoteSimulaionCheckpoint):
        assert self.current_dt < checkpoint.current_dt
        self.areas = checkpoint.areas
        self.current_dt = checkpoint.current_dt

        if self.summary_writer:
            checkpoint.write_to_tensorboard(self.summary_writer)


@dataclass
class RemoteSimulaionAgent:
    bfs: BuildingFacilitySimulator
    model: RlModel
    train_start_dt: datetime

    def simulate_and_train(self) -> RemoteSimulaionCheckpoint:
        history: list[RemoteSimulationHistory] = []

        print(f"Resume simulation from {self.bfs.get_current_datetime()}", flush=True)

        while self.bfs.get_current_datetime() < self.train_start_dt:
            history.append(self._simulate_1step(False))

        print(f"Start training from {self.bfs.get_current_datetime()}.", flush=True)

        while not self.bfs.has_finished():
            history.append(self._simulate_1step())
        
        return RemoteSimulaionCheckpoint(
            model=self.model,
            areas=self.bfs.areas,
            current_dt=self.bfs.get_current_datetime(),
            history=history
        )

    def _simulate_1step(self, train_model: bool = True) -> RemoteSimulationHistory:

        _, action, reward = self.bfs.step_with_model(self.model, train_model)
        building_action = BuildingAction.from_ndarray(action, self.bfs.areas)

        return RemoteSimulationHistory(
            steps=self.bfs.cur_steps,
            state=self.bfs.get_state(),
            reward=reward,
            action=building_action
        )


@dataclass
class RemoteSimulaionCheckpoint:
    model: RlModel
    areas: list[Area]
    current_dt: datetime
    history: list[RemoteSimulationHistory]


    def write_to_tensorboard(self, writer: SummaryWriter):
        for history in self.history:
            writer.add_scalar("reward", history.reward, history.steps)

            for area, area_state, area_action in zip(self.areas, history.state.areas, history.action.areas):

                writer.add_scalar(f"temperature_{area.name}", area_state.temperature, history.steps)
                writer.add_scalar(f"power_consumption_{area.name}", area_state.power_consumption, history.steps)


class RemoteSimulationHistory(NamedTuple):
    steps: int
    state: BuildingState
    reward: np.ndarray
    action: BuildingAction
