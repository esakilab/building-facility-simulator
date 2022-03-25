from datetime import datetime
from pathlib import Path
from threading import Thread, Condition
import time
import docker

from distributed_platform.server import FLServer
from rl.sac import SAC, average_sac

HOSTS = [
    "10.17.104.142",
    "10.17.104.143",
    "10.17.104.145",
    "10.17.104.147",
    "10.17.104.151",
    "10.17.104.152",
    "10.17.104.153",
    "10.17.104.156",
    "10.17.108.62",
    "10.17.108.65",
    "10.17.108.67",
    "10.17.108.69",
    "10.17.108.70",
    "10.17.108.71",
    "10.17.108.75",
    "10.17.108.76"
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
                environment={"GLOBAL_HOSTNAME": "10.17.104.108"},
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


def experiment(total_clients: int, nodes: int, log_file_path: Path):
    hosts = HOSTS[:nodes]

    stop_local_containers(hosts)
    build_local_containers(set(hosts) - container_built_hosts)

    while True:
        try:
            server = FLServer(SAC, datetime(2020, 8, 1), 10080, 60, 10, average_sac, device='cpu')
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

    with log_file_path.open('a') as f:
        f.write(f"{nodes}\t{elapsed_time}\n")


if __name__ == "__main__":
    for nodes in [16, 8, 4, 2, 1]:
        experiment(128, nodes, Path(f"./experimental_logfiles/7days_10nodes_per_round.tsv"))
