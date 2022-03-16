from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from threading import Thread
import time
import docker

from distributed_platform.server import FLServer
from rl.sac import SAC, average_sac

HOSTS = [
    "10.17.104.125",
    "10.17.104.126",
    "10.17.104.137",
    "10.17.104.140",
    "10.17.108.45",
    "10.17.108.46",
    "10.17.108.47",
    "10.17.108.48",
    "10.17.108.49",
    "10.17.108.50",
    "10.17.108.51",
    "10.17.108.52",
    "10.17.108.53",
    "10.17.108.54",
    "10.17.108.56",
    "10.17.108.57",
]

def build_local_container(host: str):
    cli = docker.APIClient(base_url=f"ssh://{host}")
    """
    最初 docker.errors.DockerException: Install paramiko package to enable ssh:// support
    が出たが、言われた通り conda install -c anaconda paramiko したら回避できた
    """
    cli.build(path='.', tag='local')
    print(f"Build done for {host}", flush=True)


def start_local_containers(host: str, scale: int=1):
    cli = docker.DockerClient(base_url=f"ssh://{host}")
    for _ in range(scale):
        cli.containers.run(
            image="local", 
            command="python local.py", 
            environment={"GLOBAL_HOSTNAME": "10.17.104.108"},
            detach=True)
    
    print(f"Started {scale} local containers on {host}", flush=True)


def stop_local_containers(host: str) -> list:
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



def experiment(total_clients: int, nodes: int, log_file_path: Path):
    hosts = HOSTS[:nodes]

    with Pool() as p:
        p.map(stop_local_containers, hosts)
        p.map(build_local_container, hosts)

    while True:
        try:
            server = FLServer(SAC, datetime(2020, 8, 1), 10080, 60, 4, average_sac, device='cpu')
            break
        except OSError as e:
            print(f"Failed to start server: {e}\nRetry after 3 sec...", flush=True)
            time.sleep(3)
            pass

    start_time: float = time.perf_counter()
    thread = Thread(target=server.run, daemon=True)
    thread.start()

    with Pool() as p:
        p.starmap(start_local_containers, list((host, total_clients // nodes) for host in hosts))

    thread.join()

    elapsed_time: float = time.perf_counter() - start_time

    print(f"Elapsed time: {elapsed_time}", flush=True)

    with log_file_path.open('a') as f:
        f.write(f"{nodes}\t{elapsed_time}\n")


if __name__ == "__main__":
    for nodes in [4, 2, 1]: # [16, 8, 4, 2, 1]:
        experiment(128, nodes, Path(f"./experimental_logfiles/7days_4nodes_per_round.tsv"))
