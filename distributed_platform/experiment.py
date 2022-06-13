from datetime import datetime
import os
import time
from typing import Callable, Type, TypeVar
import docker
from itertools import cycle, islice
import numpy as np
from pydantic import BaseModel, stricturl, validator

from distributed_platform.server import FLServer
from simulator.building import BuildingAction, BuildingState
from simulator.interfaces.model import RlModel

# Dockerのホストurl
# 参考： https://docs.docker.jp/engine/reference/commandline/dockerd.html#daemon-socket-option
DockerDaemonUrl = stricturl(tld_required=False, allowed_schemes={"unix", "tcp", "fd", "ssh"})

class ExperimentConfig(BaseModel):
    global_node_ip: str
    selection_port: int
    reporting_port: int
    local_node_urls: list[DockerDaemonUrl]
    total_client_num: int
    round_client_num: int
    start_time: datetime
    steps_per_round: int
    total_steps: int
    
    @validator('reporting_port')
    def check_port_conflict(cls, value, values):
        if value == values['selection_port']:
            raise ValueError(f'`selection_port` and `reporting_port` cannot have the same value ({value}).')
        return value


M = TypeVar('M', bound=RlModel)
class Experiment:
    def __init__(
            self, ModelClass: Type[M], model_aggregation: Callable[[list[M]], M], 
            calc_reward: Callable[[BuildingState, BuildingAction], np.ndarray], config: ExperimentConfig):
            
        print(f"Expermient Configuration: {config.json(indent=4, separators=(',', ': '))}", flush=True)

        self.ModelClass = ModelClass
        self.model_aggregation = model_aggregation
        self.calc_reward = calc_reward

        self.env: dict[str, str] = {
            "GLOBAL_HOSTNAME": config.global_node_ip,
            "SELECTION_PORT": str(config.selection_port),
            "REPORTING_PORT": str(config.reporting_port),
        }

        os.environ.update(self.env)

        self.docker_clients: dict[str, docker.DockerClient] = {
            str(url): docker.DockerClient(base_url=url) for url in config.local_node_urls
        }

        self.total_client_num: int = config.total_client_num
        self.round_client_num: int = config.round_client_num

        self.start_time: datetime = config.start_time
        self.steps_per_round: int = config.steps_per_round
        self.total_steps: int = config.total_steps

        self._build_local_containers()
    

    def run(self):
        server: FLServer = self._start_global_server()

        server._start_selection_thread()
        self._start_local_containers()

        server._wait_for_clients(self.total_client_num)

        start_time: float = time.perf_counter()
        server._exec_fl_proccess()
        elapsed_time: float = time.perf_counter() - start_time

        print(f"Elapsed time: {elapsed_time}", flush=True)

    
    def _build_local_containers(self):
        self._stop_local_containers()
        for url, cli in self.docker_clients.items():
            """
            最初 docker.errors.DockerException: Install paramiko package to enable ssh:// support
            が出たが、言われた通り conda install -c anaconda paramiko したら回避できた
            """
            cli.images.build(path='.', tag='bfs/local:latest', rm=True)

            print(f"Build done for {url}", flush=True)

    

    def _start_local_containers(self):
        for cli in islice(cycle(self.docker_clients.values()), self.total_client_num):
            cli.containers.run(
                image="bfs/local", 
                command="python -c 'from distributed_platform.client import FLClient; FLClient().run()'",  
                environment=self.env,
                detach=True)
        
        print(f"Started {self.total_client_num} local containers using hosts: {list(self.docker_clients.keys())}", flush=True)

    
    def _stop_local_containers(self):
        for url, cli in self.docker_clients.items():
            containers = cli.containers.list()
            if len(containers) > 0:
                print(f"Found {len(containers)} dangling local containers found on {url}.", flush=True)
                print(f"Trying to stop them...", flush=True)
                for container in cli.containers.list():
                    container.kill()
                
                print(f"Stopped all local containers on {url}", flush=True)
            else:
                print(f"No dangling local containers found on {url}", flush=True)
    
    
    def _start_global_server(self) -> FLServer:
        while True:
            try:
                return FLServer(
                    self.ModelClass, 
                    self.start_time, 
                    self.total_steps, 
                    self.steps_per_round, 
                    self.round_client_num, 
                    self.model_aggregation,
                    self.calc_reward,
                    device='cpu')

            except OSError as e:
                print(f"Failed to start server: {e}\nRetry after 5 sec...", flush=True)
                time.sleep(5)
