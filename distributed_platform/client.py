import socket
import pickle
import time
from typing import Any, Optional

import numpy as np

from distributed_platform.utils import GLOBAL_HOSTNAME, SELECTION_PORT, REPORTING_PORT, action_to_ES, action_to_temp, recv_all, send_all
from simulator.bfs import BuildingFacilitySimulator

class FLClient:
    def __init__(self):
        self.client_id: Optional[str] = None

    def run(self):
        time.sleep(1)

        while True:
            print(f"Saying hello to global..", flush=True)
            resp = self._send_request({'message': 'hello'}, SELECTION_PORT)

            # 最初のアクセスで発行される
            if 'client_id' in resp:
                self.client_id = resp['client_id']

            # TODO: 選ばれなかった場合の処理を書く
            # 選ばれなかった場合は、学習はしないが、bfsのステップは進めて状態は更新するといいかも
            model = resp['model']
            if 'simulator' in resp:
                bfs: BuildingFacilitySimulator = resp['simulator']

            print(f"Resume simulation from {bfs.get_current_datetime()}", flush=True)

            req = {
                'model': model, 
                'steps': [], 
                'states': [], 
                'rewards': [], 
                'temps': [], 
                'modes': []
            }

            while bfs.get_current_datetime() < resp['start_datetime']:
                self._simulate_1step(bfs, req, False)

            print(f"Start training from {bfs.get_current_datetime()}.", flush=True)

            while bfs.get_current_datetime() < resp['end_datetime']:
                self._simulate_1step(bfs, req)

            print("Sending local model to global..", flush=True)
            resp = self._send_request(req, REPORTING_PORT)

            if bfs.has_finished():
                break
    
    def _simulate_1step(
            self, 
            bfs: BuildingFacilitySimulator, 
            reporting_req: dict[str, Any], 
            train_model: bool = True):

        state, action, reward = bfs.step_with_model(reporting_req['model'], train_model)

        # TODO: この辺りをより柔軟にする　 & sliceのハードコーディングをやめる
        if train_model:
            reporting_req['steps'].append(bfs.cur_steps)
            reporting_req['states'].append(bfs.get_state())
            reporting_req['rewards'].append(reward)
            reporting_req['temps'].append(action_to_temp(action[1::2]))
            reporting_req['modes'].append(action_to_ES(action[-1]))
        

    def _send_request(self, payload: dict[str, Any], port) -> dict[str, Any]:
        payload['client_id'] = self.client_id

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((GLOBAL_HOSTNAME, port))
            send_all(pickle.dumps(payload), s)

            return pickle.loads(recv_all(s))
