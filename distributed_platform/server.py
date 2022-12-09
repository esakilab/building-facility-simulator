from __future__ import annotations
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from random import shuffle
import socket
import pickle
import threading
from queue import Queue
import time
from typing import Any, Callable, Optional, Type, TypeVar

import numpy as np

from distributed_platform.remote_simulation import RemoteSimulaionCheckpoint, RemoteSimulatonManager
from distributed_platform.utils import SELECTION_PORT, REPORTING_PORT, recv_all, send_all
from simulator.bfs import BuildingFacilitySimulator
from simulator.building import BuildingAction, BuildingState
from simulator.interfaces.config import SimulatorConfig
from simulator.interfaces.model import RlModel

CalcReward = Callable[[BuildingState, BuildingAction], np.ndarray]
M = TypeVar('M', bound=RlModel)

class FLServer():
    # TODO: サーバを建てるportを指定できるようにすることで、
    #     : ModelClassごとに複数建てられるようにすれば良さそう
    def __init__(
            self, 
            ModelClass: Type[M],
            start_time: datetime, 
            total_steps: int, 
            steps_per_round: int, 
            round_client_num: int, 
            model_aggregation: Callable[[list[M]], M],
            config_paths_with_tag: list[tuple[Path, str]],
            tag_to_calc_reward: dict[str, CalcReward],
            **model_constructor_kwargs):

        self.ModelClass: Type[M] = ModelClass
        self.model_constructor_kwargs: dict[str, Any] = model_constructor_kwargs
        
        self.experiment_id = str(datetime.now())
        self.cur_time: datetime = start_time
        self.end_time: datetime = start_time + timedelta(minutes=total_steps)
        self.steps_per_round: int = steps_per_round
        self.round_client_num: int = round_client_num
        self.model_aggregation: Callable[[list[M]], M] = model_aggregation

        self.managers: list[RemoteSimulatonManager] = list()
        self.client_id_to_tag: list[str] = list()
        self.config_paths_with_tag: deque[tuple[Path, str]] = deque(config_paths_with_tag)
        shuffle(self.config_paths_with_tag)
        self.tag_to_selected_client_queue: defaultdict[str, Queue[tuple[socket.socket, socket._RetAddress, dict]]] = defaultdict(Queue)
        self.tag_to_global_model: dict[str, Optional[M]] = dict()

        self.selection_socket: socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.selection_socket.bind(('0.0.0.0', SELECTION_PORT))
        self.selection_socket.listen()

        self.reporting_socket: socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.reporting_socket.bind(('0.0.0.0', REPORTING_PORT))
        self.reporting_socket.listen()

        self.tag_to_calc_reward = tag_to_calc_reward


    def run(self):
        self._start_selection_thread()
        self._exec_fl_process()
    

    def _start_selection_thread(self):
        threading.Thread(target=self.selection_phase, daemon=True).start()

        print("Started the Global Server!", flush=True)
    

    def _exec_fl_proccess(self):
        while self.cur_time < self.end_time:
            time.sleep(0.1)

            if any(qsize < self.round_client_num for qsize in self._get_all_queue_sizes()):
                continue

            print(f"\n\nSTART NEW ROUND (time: {self.cur_time}, tags: {list(self.tag_to_selected_client_queue.keys())})\n", flush=True)

            self.configuration_phase()
            self.reporting_phase()


    def selection_phase(self):
        while self.cur_time < self.end_time:
            (connection, client) = self.selection_socket.accept()
                
            req = pickle.loads(recv_all(connection))
            req['client_id'] = req['client_id'] if req['client_id'] != None else self._init_client(client)
            tag = self.client_id_to_tag[req['client_id']]

            print(f"[SELECTOR] Selected {self._to_client_str(req['client_id'], client)} for the next round.", flush=True)
            self.tag_to_selected_client_queue[tag].put((connection, client, req))
    

    def configuration_phase(self):
        print("\n### Configuration Phase ###\n", flush=True)

        end_time = self.cur_time + timedelta(minutes=self.steps_per_round)

        for _ in range(self.round_client_num):
            for tag in set(self.client_id_to_tag):
                conn, client, req = self.tag_to_selected_client_queue[tag].get()

                client_id: int = req['client_id']
                
                # TODO: 選択しなかった場合は、何分後にretryしてねという情報を入れる
                resp = dict(
                    client_id=client_id,
                    agent=self.managers[client_id].create_agent(
                        model=self.tag_to_global_model[tag], 
                        train_start_dt=self.cur_time, 
                        end_dt=end_time
                    )
                )
                
                print(f"Sending global model to {self._to_client_str(client_id, client)}...", flush=True)
                
                send_all(pickle.dumps(resp), conn)
        
        self.cur_time = end_time
    

    def reporting_phase(self):
        print("\n### Reporting Phase ###\n", flush=True)

        # TODO: 遅すぎるクライアントへの対応

        connections = []
        tag_to_models: defaultdict[str, list[RlModel]] = defaultdict(list)

        for _ in range(self.round_client_num * len(self.tag_to_global_model)):
            (connection, client) = self.reporting_socket.accept()
                
            req = pickle.loads(recv_all(connection))

            client_id = req['client_id']
            tag = self.client_id_to_tag[client_id]

            print(f"Got report from {self._to_client_str(client_id, client)}.", flush=True)

            checkpoint: RemoteSimulaionCheckpoint = req['checkpoint']
            self.managers[client_id].load_checkpoint(checkpoint)
            
            connections.append(connection)
            tag_to_models[tag].append(checkpoint.model)
        
        print("Updated global model with FedAvg!", flush=True)
        for tag, models in tag_to_models.items():
            self.tag_to_global_model[tag] = self.model_aggregation(models)
            print(f"Aggregated into global model for tag: {tag}!", flush=True)

        for conn in connections:
            send_all(pickle.dumps({'success': True}), conn)
            conn.close()
    

    def _wait_for_clients(self, client_num: int):
        print(f"Waiting for {client_num} clients to respond...", flush=True)
        
        while sum(self._get_all_queue_sizes()) < client_num:
            time.sleep(0.1)

        print(f"{sum(self._get_all_queue_sizes())} clients responded!", flush=True)
    

    def _init_client(self, client: socket._RetAddress) -> int:
        config_path, tag = self.config_paths_with_tag.popleft()
        # TODO: queueに戻すのをやめ、空になった場合にエラーを返すようにする
        self.config_paths_with_tag.append((config_path, tag))

        client_id = len(self.managers)
        self.client_id_to_tag.append(tag)
        
        config = SimulatorConfig.parse_file(config_path)
        self.managers.append(
            RemoteSimulatonManager(
                config=config,
                calc_reward=self.tag_to_calc_reward[tag],
                summary_dir=f"./logs/distributed-platform-on-cluster/{self.experiment_id}/{tag}/{config_path.stem}"
            ))

        if tag not in self.tag_to_global_model:
            self.tag_to_global_model[tag] = BuildingFacilitySimulator(config, self.tag_to_calc_reward[tag])\
                .create_rl_model(self.ModelClass, **self.model_constructor_kwargs)

        print(f"- Initialized simulator for {self._to_client_str(client_id, client)} (tag: {tag}) using {config_path}.", flush=True)

        return client_id

    
    def _to_client_str(self, client_id: str, client: socket._RetAddress) -> str:
        if client_id != None:
            return f"client{client_id}(addr: {client[0]}, tag: {self.client_id_to_tag[client_id]})"
        else:
            return f"new_client({client[0]})"

    
    def _get_all_queue_sizes(self) -> list[int]:
        return [q.qsize() for q in self.tag_to_selected_client_queue.values()]
