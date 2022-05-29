from __future__ import annotations
from csv import DictWriter
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import TextIOWrapper
from itertools import cycle, islice
from pathlib import Path
from typing import Callable, Dict, Iterator, NamedTuple, Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from distributed_platform.utils import action_to_ES, action_to_temp, write_to_tensorboard
from simulator.area import Area
from simulator.bfs import BuildingFacilitySimulator
from simulator.building import BuildingAction, BuildingState
from simulator.environment import AreaEnvironment, BuildingEnvironment, ExternalEnvironment
from simulator.interfaces.config import BuildingAttributes, SimulatorConfig
from simulator.interfaces.model import RlModel


class RemoteSimulatonManager:
    def __init__(
            self, 
            config: SimulatorConfig, 
            calc_reward: Callable[[BuildingState, BuildingAction], np.ndarray],
            summary_dir: Optional[str],
            state_log_file_path: Optional[str],
            cycle_env_iter: bool = False):
        bfs = BuildingFacilitySimulator(config=config, calc_reward=calc_reward)

        # NOTE: 実験用に、cycleをつけて無限ループ可能にしている
        self.env_iter: Iterator[BuildingEnvironment] = cycle(bfs.env_iter) if cycle_env_iter else bfs.env_iter
        self.areas: list[Area] = bfs.areas
        self.start_dt: datetime = bfs.start_time
        self.current_dt: datetime = bfs.start_time
        self.calc_reward: Callable[[BuildingState, BuildingAction], np.ndarray] = bfs.calc_reward
        self.summary_writer: Optional[SummaryWriter] = SummaryWriter(summary_dir) if summary_dir else None

        if state_log_file_path:
            self.state_writer: Optional[DictWriter] = DictWriter(
                Path(state_log_file_path).open('w'),
                fieldnames=[
                    'datetime', 'room1_temperature', 'outside_temperature', 
                    'grid_power', 'room1_power', 'solar_radiation', 'reward'],
                delimiter='\t'
            )
            self.state_writer.writeheader()
        else: 
            self.state_writer = None


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
        prev_start_dt = self.current_dt
        self.current_dt = checkpoint.current_dt

        if self.summary_writer:
            for history in checkpoint.history:
                write_to_tensorboard(
                    self.summary_writer, history.steps, history.state, 
                    history.reward, history.temp)
        
        if self.state_writer:
            for history in checkpoint.history:
                self.state_writer.writerow(
                    dict(
                        datetime=prev_start_dt + timedelta(minutes=history.steps),
                        room1_temperature=history.state.areas[1].temperature,
                        outside_temperature=history.state.temperature,
                        grid_power=history.state.power_balance,
                        room1_power=history.state.areas[1].power_consumption,
                        solar_radiation=history.state.solar_radiation,
                        reward=history.reward[0]
                    )
                )


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

        state, action, reward = self.bfs.step_with_model(self.model, train_model)

        # TODO: この辺りをより柔軟にする　 & sliceのハードコーディングをやめる
        return RemoteSimulationHistory(
            steps=self.bfs.cur_steps,
            state=self.bfs.get_state(),
            reward=reward,
            temp=action_to_temp(action[1::2]),
            # mode=action_to_ES(action[-1])
        )


@dataclass
class RemoteSimulaionCheckpoint:
    model: RlModel
    areas: list[Area]
    current_dt: datetime
    history: list[RemoteSimulationHistory]


class RemoteSimulationHistory(NamedTuple):
    steps: int
    state: BuildingState
    reward: np.ndarray
    temp: float
    # mode: str
