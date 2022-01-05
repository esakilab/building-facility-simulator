import socket
import pickle
import threading
from queue import Queue
import time

from torch.utils.tensorboard import SummaryWriter

from distributed_platform.utils import SELECTION_PORT, REPORTING_PORT, recv_all, send_all, write_to_tensorboard
from simulator.bfs import BuildingFacilitySimulator
from rl.sac import SAC

class FLServer:
    def __init__(self, round_client_num, initial_model, model_aggregation):
        self.round_client_num = round_client_num
        self.model_aggregation = model_aggregation

        self.client_writer_dict = dict()
        self.selected_client_queue = Queue()
        self.global_model = initial_model

        self.selection_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.selection_socket.bind(('0.0.0.0', SELECTION_PORT))
        self.selection_socket.listen()

        self.reporting_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.reporting_socket.bind(('0.0.0.0', REPORTING_PORT))
        self.reporting_socket.listen()


    def run(self):
        threading.Thread(target=self.selection_phase).start()

        print("Started the Global Server!", flush=True)

        while True:
            time.sleep(0.1)
            if self.selected_client_queue.qsize() < self.round_client_num:
                continue

            self.configuration_phase()
            self.reporting_phase()


    def selection_phase(self):
        while True:
            (connection, client) = self.selection_socket.accept()
                
            req = pickle.loads(recv_all(connection))

            print(f"[SELECTOR] Selected {client[0]} for the next round.", flush=True)
            self.selected_client_queue.put((connection, client, req))
    

    def configuration_phase(self):
        print("\n### Configuration Phase ###\n", flush=True)

        for _ in range(self.round_client_num):
            # 現状reqはは使っていない
            conn, client, _ = self.selected_client_queue.get()
            
            print(f"Sending global model to {client[0]}...", flush=True)

            if client[0] not in self.client_writer_dict:
                resp =  {
                    'model': self.global_model,
                    'simulator': self._init_client(client),
                }
            else:
                resp = {
                    'model': self.global_model,
                }
            
            send_all(pickle.dumps(resp), conn)
    

    def reporting_phase(self):
        print("\n### Reporting Phase ###\n", flush=True)

        # TODO: 遅すぎるクライアントへの対応

        connections = []
        models = []

        for _ in range(self.round_client_num):
            (connection, client) = self.reporting_socket.accept()
                
            req = pickle.loads(recv_all(connection))

            print(f"Got report from {client[0]}.", flush=True)

            for step, state, reward, temp, mode in \
                    zip(req['steps'], req['states'], req['rewards'], req['temps'], req['modes']):
                write_to_tensorboard(self.client_writer_dict[client[0]], step, state, reward, temp, mode)
            
            connections.append(connection)
            models.append(req["model"])
        
        print("Updated global model with FedAvg!", flush=True)
        self.global_model = self.model_aggregation(models)

        for conn in connections:
            send_all(pickle.dumps({'success': True}), conn)
            conn.close()
    

    def _init_client(self, client):
        config_path = f"./input_xmls/BFS_{len(self.client_writer_dict):02}.xml"
        self.client_writer_dict[client[0]] = SummaryWriter(log_dir=f"./logs/distributed-platform-on-cluster/{client[0]}")

        print(f"- Initialized simulator for {client[0]} using {config_path}.", flush=True)

        return BuildingFacilitySimulator(cfg_path=config_path)
