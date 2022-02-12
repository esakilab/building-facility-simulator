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

            print(f"[SELECTOR] Selected {self._to_client_str(req['client_id'], client)} for the next round.", flush=True)
            self.selected_client_queue.put((connection, client, req))
    

    def configuration_phase(self):
        print("\n### Configuration Phase ###\n", flush=True)

        for _ in range(self.round_client_num):
            # 現状reqはは使っていない
            conn, client, req = self.selected_client_queue.get()

            client_id = req['client_id']

            resp = {
                'model': self.global_model,
            }

            if client_id == None:
                client_id, bfs = self._init_client(client)
                resp.update({
                    'client_id': client_id,
                    'simulator': bfs
                })
            
            print(f"Sending global model to {self._to_client_str(client_id, client)}...", flush=True)
            
            send_all(pickle.dumps(resp), conn)
    

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

            for step, state, reward, temp, mode in \
                    zip(req['steps'], req['states'], req['rewards'], req['temps'], req['modes']):
                write_to_tensorboard(self.client_writer_dict[client_id], step, state, reward, temp, mode)
            
            connections.append(connection)
            models.append(req["model"])
        
        print("Updated global model with FedAvg!", flush=True)
        self.global_model = self.model_aggregation(models)

        for conn in connections:
            send_all(pickle.dumps({'success': True}), conn)
            conn.close()
    

    def _init_client(self, client):
        client_id = len(self.client_writer_dict)
        config_path = f"./input_xmls/BFS_{client_id:02}.xml"
        self.client_writer_dict[client_id] = SummaryWriter(log_dir=f"./logs/distributed-platform-on-cluster/{client_id}")

        print(f"- Initialized simulator for {self._to_client_str(client_id, client)} using {config_path}.", flush=True)

        return client_id, BuildingFacilitySimulator(cfg_path=config_path)

    
    def _to_client_str(self, client_id, client):
        if client_id != None:
            return f"client{client_id}({client[0]})"
        else:
            return f"new_client({client[0]})"
