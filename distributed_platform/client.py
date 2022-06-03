import socket
import pickle
import time

import numpy as np

from distributed_platform.utils import GLOBAL_HOSTNAME, ACTION_SHAPE, SELECTION_PORT, REPORTING_PORT, STATE_SHAPE, action_to_ES, action_to_temp, cvt_state_to_ndarray, recv_all, send_all
from simulator.io import BuildingAction

class FLClient:
    def __init__(self):
        self.client_id = None

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
                bfs = resp['simulator']
            
            action_arr = np.zeros(*ACTION_SHAPE)

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
    
    def _simulate_1step(self, bfs, action_arr, reporting_req, train_model=True):
        (state, reward) = bfs.step(self._action_arr_to_action(action_arr))

        state_arr = cvt_state_to_ndarray(state, bfs.cur_steps)
        reward_val = reward.metric1

        if bfs.cur_steps == 0:
            return action_arr
        elif train_model:
            reporting_req['model'].replay_buffer.add(state_arr, action_arr, state_arr, reward_val, done=False)

        action_arr, _ = reporting_req['model'].choose_action(state_arr)

        if train_model:
            reporting_req['steps'].append(bfs.cur_steps)
            reporting_req['states'].append(state)
            reporting_req['rewards'].append(reward)
            reporting_req['temps'].append(action_to_temp(action_arr[:-1]))
            reporting_req['modes'].append(action_to_ES(action_arr[-1]))

            reporting_req['model'].update()

        return action_arr
    

    # TODO: シミュレータ側で変換してから返す
    def _action_arr_to_action(self, action_arr):
        temp = action_to_temp(action_arr[:-1])  # 一番最後のESは除く
        mode = action_to_ES(action_arr[-1])

        action = BuildingAction()
        action.add(area_id=1, facility_id=0, status=True, temperature=temp[0])
        action.add(area_id=2, facility_id=0, status=True, temperature=temp[1])
        action.add(area_id=3, facility_id=0, status=True, temperature=temp[2])
        action.add(area_id=4, facility_id=0, mode=mode)

        return action
        

    def _send_request(self, payload: dict, port):
        payload['client_id'] = self.client_id

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((GLOBAL_HOSTNAME, port))
            send_all(pickle.dumps(payload), s)

            return pickle.loads(recv_all(s))
