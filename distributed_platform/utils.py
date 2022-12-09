import os
import numpy as np

from simulator.building import BuildingAction, BuildingState

BUFF_SIZE = 65536
GLOBAL_HOSTNAME = os.environ.get("GLOBAL_HOSTNAME", 'global')
SELECTION_PORT = int(os.environ.get("SELECTION_PORT", '11113'))
REPORTING_PORT = int(os.environ.get("REPORTING_PORT", '11114'))

def recv_all(conn):
    data_len = int.from_bytes(conn.recv(4), 'little')

    data = bytearray()
    while data_len > 0:
        packet = conn.recv(BUFF_SIZE)
        data += packet
        data_len -= len(packet)
    
    return data


def send_all(data, conn):
    conn.send(len(data).to_bytes(4, 'little'))
    conn.sendall(data)


def calc_reward(state: BuildingState, action: BuildingAction) -> np.ndarray:
    LAMBDA1 = 0.2
    LAMBDA2 = 0.1
    LAMBDA3 = 0.01
    LAMBDA4 = 20
    T_MAX = 30
    T_MIN = 20
    T_TARGET = 25

    area_temp = np.array([area.temperature for area in state.areas])
    # area_temp = state.areas[1].temperature
    
    reward = np.exp(-LAMBDA1 * (area_temp - T_TARGET) ** 2).sum()
    reward += - LAMBDA2 * (np.where((T_MIN - area_temp) < 0, 0, (T_MIN - area_temp)).sum())
    reward += - LAMBDA2 * (np.where((area_temp - T_MAX) < 0, 0, (area_temp - T_MAX)).sum())
    reward += - LAMBDA3 * state.electric_price_unit * state.power_balance
    reward += LAMBDA4 * state.areas[4].facilities[0].charge_ratio

    return np.array([reward])
