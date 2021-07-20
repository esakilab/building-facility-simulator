from time import sleep

from src.area import Area
from src.bfs import BuildingFacilitySimulator
from src.facility.electric_storage import ESMode
from src.io import AreaState, BuildingAction

import math
import numpy as np
import torch
import sac
# from torch.utils.tensorboard import SummaryWriter

# とりあえず変数を定義
lambda1 = 1
lambda2 = 0.1
lambda3 = 0.1
lambda4 = 20
T_max = 27
T_min = 23
T_target = 25
state_shape = (17,)  # エリアごとに順に1+5+5+5+1
action_shape = (4,)  # 各HVACの制御(3つ) + Electric Storageの制御(1つ)


def cvt_state_to_ndarray(state):
    state_arr = []
    for area_id, area_state in enumerate(state.areas):
        # 状態を獲得
        state_arr.extend([
            area_state.people, 
            area_state.temperature, 
            area_state.power_consumption
        ])

        if area_id == 4:
            state_arr.append(area_state.facilities[0].charge_ratio)

    price = state.electric_price_unit

    state_arr.append(price)
    
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


if __name__ == "__main__":
    #writer = SummaryWriter(log_dir="./logs")
    bfs = BuildingFacilitySimulator("BFS_environment.xml")

    action = BuildingAction()
    action.add(area_id=1, facility_id=0, status=True, temperature=22)
    action.add(area_id=2, facility_id=0, status=True, temperature=25)
    action.add(area_id=3, facility_id=0, status=True, temperature=28)
    action.add(area_id=4, facility_id=0, mode="charge")

    # 強化学習を行うエージェントを作成 (Soft-Actor-Critic という手法を仮に用いている)

    Agent = sac.SAC(state_shape=state_shape,
                    action_shape=action_shape, device='cpu')

    # ここはとりあえず状態, 行動, 報酬, 設定温度の変数を初期化
    state = np.zeros(*state_shape)
    next_state = np.zeros(*state_shape)
    action_ = np.zeros(*action_shape)
    reward = np.zeros(1)
    temp = np.zeros(3)
    charge_ratio = 0

    for i, (state_obj, reward_obj) in enumerate(bfs.step(action)):
        next_state = cvt_state_to_ndarray(state_obj)
        reward = reward_obj.metric1

        if i >= 1:
            Agent.replay_buffer.add(
                state, action_, next_state, reward, done=False)
        state = next_state

        if i == 0:
            continue

        if i >= 100:
            action_, _ = Agent.choose_action(state)
        else:
            action_ = np.random.uniform(low=-1, high=1, size=4)

        temp = action_to_temp(action_[:-1])  # 一番最後のESは除く
        mode = action_to_ES(action_[-1])
        action.add(area_id=1, facility_id=0, status=True, temperature=temp[0])
        action.add(area_id=2, facility_id=0, status=True, temperature=temp[1])
        action.add(area_id=3, facility_id=0, status=True, temperature=temp[2])
        action.add(area_id=4, facility_id=0, mode=mode)
        '''
        writer.add_scalar("set_temperature_area1", temp[0], i)
        writer.add_scalar("set_temperature_area2", temp[1], i)
        writer.add_scalar("set_temperature_area3", temp[2], i)
        writer.add_scalar(
            "temperature_area1", building_state.area_states[1].temperature, i)
        writer.add_scalar(
            "temperature_area2", building_state.area_states[2].temperature, i)
        writer.add_scalar(
            "temperature_area3", building_state.area_states[3].temperature, i)
        if mode == 'charge':
            mode_ = 1
        elif mode == 'stand_by':
            mode_ = 0
        else:
            mode_ = -1
        # writer.add_scalar('charge_mode_per_price', price, mode_)
        writer.add_scalar('charge_mode_per_time', mode_, i)
        writer.add_scalar('charge_ratio', area.facilities[0].charge_ratio, i)
        '''
        Agent.update()

        if i % 60 == 0:
            bfs.print_cur_state()
