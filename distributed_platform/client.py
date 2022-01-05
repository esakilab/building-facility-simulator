import socket
import pickle
import time

import numpy as np

from distributed_platform.utils import GLOBAL_HOSTNAME, ACTION_SHAPE, SELECTION_PORT, REPORTING_PORT, STATE_SHAPE, action_to_ES, action_to_temp, cvt_state_to_ndarray, recv_all, send_all
from rl.sac import average_sac
from simulator.io import BuildingAction

class FLClient:

    def run(self):
        time.sleep(1)

        action = BuildingAction()

        while True:
            print(f"Saying hello to global..", flush=True)
            resp = self._send_request({'message': 'hello'}, SELECTION_PORT)

            # TODO: 選ばれなかった場合の処理を書く
            # 選ばれなかった場合は、学習はしないが、bfsのステップは進めて状態は更新するといいかも
            sac = resp['model']
            if 'simulator' in resp:
                bfs = resp['simulator']
            
            state = cvt_state_to_ndarray(bfs.get_state(), bfs.cur_steps)
            action_arr = np.zeros(*ACTION_SHAPE)
            
            steps = []
            states = []
            rewards = []
            temps = []
            modes = []

            print(f"Resume training from step {bfs.cur_steps}", flush=True)

            for _ in range(60):
                (state_obj, reward_obj) = bfs.step(action)

                next_state = cvt_state_to_ndarray(state_obj, bfs.cur_steps)
                reward = reward_obj.metric1

                if bfs.cur_steps >= 1:
                    sac.replay_buffer.add(state, action_arr, next_state, reward, done=False)

                state = next_state

                if bfs.cur_steps == 0:
                    continue

                if bfs.cur_steps >= 100:
                    action_arr, _ = sac.choose_action(state)
                else:
                    action_arr = np.random.uniform(low=-1, high=1, size=4)

                temp = action_to_temp(action_arr[:-1])  # 一番最後のESは除く
                mode = action_to_ES(action_arr[-1])

                action = BuildingAction()
                action.add(area_id=1, facility_id=0, status=True, temperature=temp[0])
                action.add(area_id=2, facility_id=0, status=True, temperature=temp[1])
                action.add(area_id=3, facility_id=0, status=True, temperature=temp[2])
                action.add(area_id=4, facility_id=0, mode=mode)


                steps.append(bfs.cur_steps)
                states.append(state_obj)
                rewards.append(reward_obj)
                temps.append(temp)
                modes.append(mode)

                sac.update()
            
            print("Sending local model to global..", flush=True)
            resp = self._send_request({
                'model': sac, 
                'steps': steps, 
                'states': states, 
                'rewards': rewards, 
                'temps': temps, 
                'modes': modes
            }, REPORTING_PORT)

            if bfs.has_finished():
                break
        

    def _send_request(self, payload: dict, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((GLOBAL_HOSTNAME, port))
            send_all(pickle.dumps(payload), s)

            return pickle.loads(recv_all(s))
