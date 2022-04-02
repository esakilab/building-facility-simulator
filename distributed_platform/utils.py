import os
import numpy as np

from simulator.building import BuildingAction, BuildingState

GLOBAL_HOSTNAME = os.environ.get("GLOBAL_HOSTNAME", 'global')
BUFF_SIZE = 65536
SELECTION_PORT = 11113
REPORTING_PORT = 11114

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


def action_to_temp(action):
    # action = [-1,1] -> temp = [15, 30]

    temp = action * 7.5 + 22.5
    return np.floor(temp)


def action_to_ES(action):

    if - 1 <= action < -1 / 3:
        mode = 'charge'
    elif - 1 / 3 <= action <= 1 / 3:
        mode = 'stand_by'
    else:
        mode = 'discharge'
    return mode


def write_to_tensorboard(writer, step, state_obj, reward_arr, temp):
        
    writer.add_scalar("reward", reward_arr[0], step)

    writer.add_scalar("set_temperature_area1", temp[0], step)
    # writer.add_scalar("set_temperature_area2", temp[1], step)
    # writer.add_scalar("set_temperature_area3", temp[2], step)
    writer.add_scalar(
        "temperature_area1", state_obj.areas[1].temperature, step)
    # writer.add_scalar(
    #     "temperature_area2", state_obj.areas[2].temperature, step)
    # writer.add_scalar(
    #     "temperature_area3", state_obj.areas[3].temperature, step)
    
    # mode_dict = {
    #     'charge': 1,
    #     'stand_by': 0,
    #     'discharge': -1
    # }
    # # writer.add_scalar('charge_mode_per_price', price, mode_)
    # writer.add_scalar('charge_mode_per_time', mode_dict[mode], step)
    # writer.add_scalar('charge_ratio', state_obj.areas[4].facilities[0].charge_ratio, step)


def calc_reward(state: BuildingState, action: BuildingAction) -> np.ndarray:
    LAMBDA1 = 0.2
    LAMBDA2 = 0.1
    LAMBDA3 = 0.01
    LAMBDA4 = 20
    T_MAX = 30
    T_MIN = 20
    T_TARGET = 25

    # area_temp = np.array([area.temperature for area in state.areas])
    area_temp = state.areas[1].temperature
    
    reward = np.exp(-LAMBDA1 * (area_temp - T_TARGET) ** 2).sum()
    # reward += - LAMBDA2 * (np.where((T_MIN - area_temp) < 0, 0, (T_MIN - area_temp)).sum())
    # reward += - LAMBDA2 * (np.where((area_temp - T_MAX) < 0, 0, (area_temp - T_MAX)).sum())
    # reward += - LAMBDA3 * state.electric_price_unit * state.power_balance
    # reward += LAMBDA4 * state.areas[4].facilities[0].charge_ratio

    return np.array([reward])
