from __future__ import annotations
from datetime import datetime, timedelta
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
            calc_reward: Callable[[BuildingState, BuildingAction], np.ndarray],
            **model_constructor_kwargs):

        self.ModelClass: Type[M] = ModelClass
        self.model_constructor_kwargs: dict[str, Any] = model_constructor_kwargs
        
        self.experiment_id = str(datetime.now())
        self.cur_time: datetime = start_time
        self.end_time: datetime = start_time + timedelta(minutes=total_steps)
        self.steps_per_round: int = steps_per_round
        self.round_client_num: int = round_client_num
        self.model_aggregation: Callable[[list[M]], M] = model_aggregation

        self.manager_dict: dict[str, RemoteSimulatonManager] = dict()
        self.selected_client_queue: Queue[tuple[socket.socket, socket._RetAddress, dict]] = Queue()
        self.global_model: Optional[M] = None

        self.selection_socket: socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.selection_socket.bind(('0.0.0.0', SELECTION_PORT))
        self.selection_socket.listen()

        self.reporting_socket: socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.reporting_socket.bind(('0.0.0.0', REPORTING_PORT))
        self.reporting_socket.listen()

        self.calc_reward = calc_reward


    def run(self):
        self._start_selection_thread()
        self._exec_fl_process()
    

    def _start_selection_thread(self):
        threading.Thread(target=self.selection_phase, daemon=True).start()

        print("Started the Global Server!", flush=True)
    

    def _exec_fl_proccess(self):
        while self.cur_time < self.end_time:
            time.sleep(0.1)
            if self.selected_client_queue.qsize() < self.round_client_num:
                continue

            print(f"\n\nSTART NEW ROUND (time: {self.cur_time}, qsize: {self.selected_client_queue.qsize()})\n", flush=True)

            self.configuration_phase()
            self.reporting_phase()


    def selection_phase(self):
        while self.cur_time < self.end_time:
            (connection, client) = self.selection_socket.accept()
                
            req = pickle.loads(recv_all(connection))

            print(f"[SELECTOR] Selected {self._to_client_str(req['client_id'], client)} for the next round.", flush=True)
            self.selected_client_queue.put((connection, client, req))
    

    def configuration_phase(self):
        print("\n### Configuration Phase ###\n", flush=True)

        end_time = self.cur_time + timedelta(minutes=self.steps_per_round)

        for _ in range(self.round_client_num):
            conn, client, req = self.selected_client_queue.get()

            client_id = req['client_id'] if req['client_id'] != None else self._init_client(client)
            
            # TODO: 選択しなかった場合は、何分後にretryしてねという情報を入れる
            resp = dict(
                client_id=client_id,
                agent=self.manager_dict[client_id].create_agent(self.global_model, self.cur_time, end_time)
            )
            
            print(f"Sending global model to {self._to_client_str(client_id, client)}...", flush=True)
            
            send_all(pickle.dumps(resp), conn)
        
        self.cur_time = end_time
    

    def reporting_phase(self):
        print("\n### Reporting Phase ###\n", flush=True)

        # TODO: 遅すぎるクライアントへの対応

        connections = []
        models = []

        for _ in range(self.round_client_num):
            (connection, client) = self.reporting_socket.accept()
                
            req = pickle.loads(recv_all(connection))

            client_id = req['client_id']

            print(f"Got report from {self._to_client_str(client_id, client)}.", flush=True)

            checkpoint: RemoteSimulaionCheckpoint = req['checkpoint']
            self.manager_dict[client_id].load_checkpoint(checkpoint)
            
            connections.append(connection)
            models.append(checkpoint.model)
        
        print("Updated global model with FedAvg!", flush=True)
        self.global_model = self.model_aggregation(models)

        for conn in connections:
            send_all(pickle.dumps({'success': True}), conn)
            conn.close()
    

    def _wait_for_clients(self, client_num: int):
        print(f"Waiting for {client_num} clients to respond...", flush=True)
        
        while self.selected_client_queue.qsize() < client_num:
            time.sleep(0.1)

        print(f"{self.selected_client_queue.qsize()} clients responded!", flush=True)
    

    def _init_client(self, client: socket._RetAddress) -> str:
        client_id = len(self.manager_dict)
        config_path = f"./data/json/BFS_{client_id:02}/simulator_config.json"
        config = SimulatorConfig.parse_file(config_path)
        self.manager_dict[client_id] = RemoteSimulatonManager(
            config=config,
            calc_reward=self.calc_reward,
            summary_dir=f"./logs/distributed-platform-on-cluster/{self.experiment_id}/{client_id}"
        )

        if self.global_model is None:
            self.global_model = BuildingFacilitySimulator(config, self.calc_reward)\
                .create_rl_model(self.ModelClass, **self.model_constructor_kwargs)

        print(f"- Initialized simulator for {self._to_client_str(client_id, client)} using {config_path}.", flush=True)

        return client_id

    
    def _to_client_str(self, client_id: str, client: socket._RetAddress) -> str:
        if client_id != None:
            return f"client{client_id}({client[0]})"
        else:
            return f"new_client({client[0]})"
