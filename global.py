from distributed_platform.server import FLServer
from distributed_platform.utils import ACTION_SHAPE, STATE_SHAPE
from rl.sac import SAC, average_sac


if __name__ == "__main__":
    server = FLServer(8, SAC(state_shape=STATE_SHAPE, action_shape=ACTION_SHAPE, device="cpu"), average_sac)
    server.run()
