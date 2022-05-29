from datetime import datetime
import gc
from pathlib import Path
from threading import Thread, Condition
import time
# import tracemalloc
from typing import Optional
import docker

from distributed_platform.server import FLServer
from rl.sac import SAC, average_sac

HOSTS = [
    "10.17.104.164",
    "10.17.104.165",
    "10.17.104.166",
    "10.17.104.167",
    "10.17.104.168",
    "10.17.104.170",
    "10.17.104.175",
    "10.17.104.177",
    "10.17.104.178",
    "10.17.104.179",
    "10.17.108.80",
    "10.17.108.81",
    "10.17.108.82",
    "10.17.108.83",
    "10.17.108.84",
    "10.17.108.85",
    "10.17.108.86",
    "10.17.108.87",
    "10.17.108.88",
    "10.17.108.90",
    "10.17.108.91",
    "10.17.108.92",
    "10.17.108.93",
    "10.17.108.94",
    "10.17.108.95",
    "10.17.108.96",
    "10.17.108.97",
    "10.17.108.98",
    "10.17.108.99",
    "10.17.108.100",
    "10.17.108.101",
    "10.17.108.102",
]

container_built_hosts: set[str] = set()

def build_local_containers(hosts: set[str]):
    for host in hosts:
        cli = docker.DockerClient(base_url=f"ssh://{host}")
        """
        最初 docker.errors.DockerException: Install paramiko package to enable ssh:// support
        が出たが、言われた通り conda install -c anaconda paramiko したら回避できた
        """
        cli.images.build(path='.', tag='bfs/local:latest', rm=True)

        container_built_hosts.add(host)
        print(f"Build done for {host}", flush=True)


def start_local_containers(hosts: list[str], scale: int=1):
    clis = [docker.DockerClient(base_url=f"ssh://{host}") for host in hosts]

    for _ in range(scale):
        for cli in clis:
            cli.containers.run(
                image="bfs/local", 
                command="python local.py", 
                environment={"GLOBAL_HOSTNAME": "10.17.104.157"},
                detach=True)
    
    print(f"Started {scale} local containers on each host in: {hosts}", flush=True)


def stop_local_containers(hosts: list[str]) -> list:
    for host in hosts:
        cli = docker.DockerClient(base_url=f"ssh://{host}")
        containers = cli.containers.list()
        if len(containers) > 0:
            print(f"Found {len(containers)} dangling local containers found on {host}.", flush=True)
            print(f"Trying to stop them...", flush=True)
            for container in cli.containers.list():
                container.stop()
            
            print(f"Stopped all local containers on {host}", flush=True)
        else:
            print(f"No dangling local containers found on {host}", flush=True)


def wait_for_all_clients_to_be_ready(server: FLServer, client_num: int):
    print("Waiting for all the clients to respond...", flush=True)
    
    while server.selected_client_queue.qsize() < client_num:
        time.sleep(0.1)

    print(f"{server.selected_client_queue.qsize()} clients responded!", flush=True)


def experiment(
        total_clients: int, 
        nodes: int,
        total_steps: int,
        round_client_num: int,
        log_file_path: Optional[Path] = None, 
        log_states: bool = False,
        cycle_env_iter: bool = False):

    hosts = HOSTS[:nodes]

    stop_local_containers(hosts)
    build_local_containers(set(hosts) - container_built_hosts)

    while True:
        try:
            server = FLServer(
                SAC, 
                datetime(2020, 8, 1), 
                total_steps, 
                60, 
                round_client_num, 
                average_sac, 
                log_states=log_states, 
                write_to_tensorboard=False, 
                cycle_env_iter=cycle_env_iter,
                device='cpu')
            break
        except OSError as e:
            print(f"Failed to start server: {e}\nRetry after 5 sec...", flush=True)
            time.sleep(5)
            pass
    
    server._start_selection_thread()
    start_local_containers(hosts, total_clients // nodes)

    wait_for_all_clients_to_be_ready(server, total_clients)

    start_time: float = time.perf_counter()
    
    server._exec_fl_proccess()

    elapsed_time: float = time.perf_counter() - start_time

    print(f"Elapsed time: {elapsed_time}", flush=True)

    if log_file_path:
        with log_file_path.open('a') as f:
            f.write(f"{nodes}\t{elapsed_time}\n")
    
    gc.collect()


if __name__ == "__main__":
    # tracemalloc.start(25)
    # snapshot0 = tracemalloc.take_snapshot()

    for nodes in [32, 16, 8, 4, 2, 1]:
        experiment(
            total_clients=128, 
            nodes=nodes, 
            total_steps=10080, 
            round_client_num=10, 
            log_file_path=Path(f"./experimental_logfiles/1024clis_7days_10npr.tsv"))
        
        # snapshot1 = tracemalloc.take_snapshot()

        # top_stats = snapshot1.compare_to(snapshot0, 'traceback')

        # print("[ Top 10 differences ]")
        # for stat in top_stats[:5]:
        #     print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
        #     for line in stat.traceback.format():
        #         print(line)
        # print(f"Traced memory: {tracemalloc.get_traced_memory()}")
        # break
