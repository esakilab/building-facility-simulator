from datetime import datetime
from distributed_platform.server import FLServer
from rl.sac import SAC, average_sac


if __name__ == "__main__":
    server = FLServer(SAC, datetime(2020, 8, 1), 60, 4, average_sac, device='cpu')
    server.run()
