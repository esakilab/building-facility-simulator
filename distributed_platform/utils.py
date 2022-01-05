import os
import numpy as np

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

STATE_SHAPE = (4,)
ACTION_SHAPE = (4,)

def cvt_state_to_ndarray(state, step):
    state_arr = []
    for area_id, area_state in enumerate(state.areas):
        # 状態を獲得
        if area_id == 1:
            state_arr.extend([
                # area_state.people, 
                area_state.temperature, 
                # area_state.power_consumption
            ])

    #     if area_id == 4:
    #         state_arr.append(area_state.facilities[0].charge_ratio)
    # price = state.electric_price_unit

    # state_arr.append(price)

    state_arr.append(state.temperature)
    state_arr.append(state.solar_radiation)
    state_arr.append(step % (60 * 24)) # time
    # state_arr.append((step // (60 * 24)) % 7) # day of week
    
    return np.array(state_arr)

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


def write_to_tensorboard(writer, step, state_obj, reward_obj, temp, mode):
        
    writer.add_scalar("reward", reward_obj.metric1, step)

    writer.add_scalar("set_temperature_area1", temp[0], step)
    writer.add_scalar("set_temperature_area2", temp[1], step)
    writer.add_scalar("set_temperature_area3", temp[2], step)
    writer.add_scalar(
        "temperature_area1", state_obj.areas[1].temperature, step)
    writer.add_scalar(
        "temperature_area2", state_obj.areas[2].temperature, step)
    writer.add_scalar(
        "temperature_area3", state_obj.areas[3].temperature, step)
    
    mode_dict = {
        'charge': 1,
        'stand_by': 0,
        'discharge': -1
    }
    # writer.add_scalar('charge_mode_per_price', price, mode_)
    writer.add_scalar('charge_mode_per_time', mode_dict[mode], step)
    writer.add_scalar('charge_ratio', state_obj.areas[4].facilities[0].charge_ratio, step)
