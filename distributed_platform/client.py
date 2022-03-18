import socket
import pickle
import time
from typing import Any, Optional

from distributed_platform.remote_simulation import RemoteSimulaionAgent
from distributed_platform.utils import GLOBAL_HOSTNAME, SELECTION_PORT, REPORTING_PORT, recv_all, send_all

class FLClient:
    def __init__(self):
        self.client_id: Optional[str] = None

    def run(self):
        time.sleep(1)

        while True:
            print(f"Saying hello to global..", flush=True)
            resp = self._send_request({'message': 'hello'}, SELECTION_PORT)

            # TODO: 終了のお知らせを受信したらbreakする

            # 最初のアクセスで発行される
            if self.client_id is None:
                self.client_id = resp['client_id']

            # TODO: 選ばれなかった場合の処理を書く
            # 選ばれなかった場合は、学習はしないが、bfsのステップは進めて状態は更新するといいかも
            agent: RemoteSimulaionAgent = resp['agent']
            checkpoint = agent.simulate_and_train()

            print("Sending local model to global..", flush=True)
            resp = self._send_request(dict(checkpoint=checkpoint), REPORTING_PORT)
        

    def _send_request(self, payload: dict[str, Any], port) -> dict[str, Any]:
        payload['client_id'] = self.client_id

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((GLOBAL_HOSTNAME, port))
            send_all(pickle.dumps(payload), s)

            return pickle.loads(recv_all(s))
