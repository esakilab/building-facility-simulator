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
            
            action_arr = np.zeros(*bfs.get_action_shape())

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
                action_arr = self._simulate_1step(bfs, action_arr, req, False)

            print(f"Start training from {bfs.get_current_datetime()}.", flush=True)

            while bfs.get_current_datetime() < resp['end_datetime']:
                action_arr = self._simulate_1step(bfs, action_arr, req)

            print("Sending local model to global..", flush=True)
            resp = self._send_request(req, REPORTING_PORT)

            if bfs.has_finished():
                break
    
    def _simulate_1step(
            self, 
            bfs: BuildingFacilitySimulator, 
            action_arr: np.ndarray, 
            reporting_req: dict[str, Any], 
            train_model: bool = True) -> np.ndarray:

        (state_arr, reward) = bfs.step(action_arr)

        if bfs.cur_steps == 0:
            return action_arr
        elif train_model:
            reporting_req['model'].replay_buffer.add(state_arr, action_arr, state_arr, reward[0], done=False)

        action_arr, _ = reporting_req['model'].choose_action(state_arr)

        # TODO: この辺りをより柔軟にする　 & sliceのハードコーディングをやめる
        if train_model:
            reporting_req['steps'].append(bfs.cur_steps)
            reporting_req['states'].append(bfs.get_state())
            reporting_req['rewards'].append(reward)
            reporting_req['temps'].append(action_to_temp(action_arr[1::2]))
            reporting_req['modes'].append(action_to_ES(action_arr[-1]))

            reporting_req['model'].update()

        return action_arr
        

    def _send_request(self, payload: dict[str, Any], port) -> dict[str, Any]:
        payload['client_id'] = self.client_id

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((GLOBAL_HOSTNAME, port))
            send_all(pickle.dumps(payload), s)

            return pickle.loads(recv_all(s))
