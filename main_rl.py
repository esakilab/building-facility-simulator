from time import sleep

from src.area import Area
from src.bfs import BuildingFacilitySimulator
from src.facility.electric_storage import ESMode
from src.io import AreaState, BuildingAction

import math
import numpy as np
import torch
import sac

# とりあえず変数を定義
lambda1 = 0.2
lambda2 = 0.1
lambda3 = 0.1
T_max = 30
T_min = 20
T_target = 25
state_shape = (3,)
action_shape = (1,)


def get_reward(temp: float, electric_price_unit: float):
    R = np.exp(-lambda1 * (temp - T_target)**2) - lambda2 * \
        (max(0, T_min - temp) + max(0, temp - T_max)) - \
        lambda3 * electric_price_unit
# 人数の項を考える
# 太陽光の発電状況
# 蓄電池の残量
# 異なるrewardを考えた状況設定
    return R


def action_to_temp(action: float):
    # action = [-1,1] -> temp = [15, 30]
    temp = action * 7.5 + 22.5
    return math.floor(temp)


def print_area(area_id: str, area: Area, area_state: AreaState):
    print(
        f"area {area_id}: temp={area.temperature:.2f}, power={area_state.power_consumption:.2f}, {area.facilities[0]}")


if __name__ == "__main__":
    bfs = BuildingFacilitySimulator("BFS_environment.xml")

    action = BuildingAction()
    action.add(area_id=1, facility_id=0, status=True, temperature=22)
    action.add(area_id=2, facility_id=0, status=True, temperature=25)
    action.add(area_id=3, facility_id=0, status=True, temperature=28)
    action.add(area_id=4, facility_id=0, mode="charge")
    # 強化学習を行うエージェントを作成 (Soft-Actor-Critic という手法を仮に用いている)
    Agent = []
    for _ in range(3):
        Agent.append(sac.SAC(state_shape=state_shape,
                             action_shape=action_shape,  device='cpu'))
    # ここはとりあえず状態, 行動, 報酬, 設定温度の変数を初期化
    states = np.zeros((3, *state_shape))
    next_states = np.zeros((3, *state_shape))
    actions = np.zeros((3, 1))
    rewards = np.zeros((3, 1))
    temp = np.zeros((3, 1))

    for i, (building_state, reward) in enumerate(bfs.step(action)):
        sleep(0.1)
        print(f"\niteration {i}")
        print(bfs.ext_envs[i])
        for area_id, area in enumerate(bfs.areas):

            print_area(area_id, area, building_state.area_states[area_id])
            # 一旦エアコン制御だけなので idが1から3で状態は各温度 電力消費量, 電力単価の3つのみで行なっている
            if 1 <= area_id <= 3:

                power_consumption = building_state.area_states[area_id].power_consumption
                next_states[area_id - 1] = np.array([area.temperature,
                                                     power_consumption, bfs.ext_envs[i].electric_price_unit])

                # ここも一旦自作関数で報酬を獲得
                rewards[area_id - 1] = get_reward(
                    area.temperature, bfs.ext_envs[i].electric_price_unit)

                # 強化学習では過去のデータを記憶させて学習させるため過去データを保存
                Agent[area_id - 1].replay_buffer.add(
                    states[area_id - 1], actions[area_id - 1], next_states[area_id - 1], rewards[area_id - 1], False)
                states[area_id - 1] = next_states[area_id - 1]

                # 各areaごとに Agent.exploreでエアコンの設定温度を決める
                if i >= 1000:
                    actions[area_id - 1], _ = Agent[area_id -
                                                    1].explore(states[area_id - 1])
                else:
                    # 最初の1000ステップはランダムに制御
                    actions[area_id - 1] = np.random.uniform(low=-1, high=1)

                # Agent.exploreの出力が-1から1なのでそれを設定温度領域に変換する
                temp[area_id - 1] = action_to_temp(actions[area_id - 1])
        # actionに設定温度を変えて入力
        for id in range(3):
            action.add(area_id=id+1, facility_id=0,
                       status=True, temperature=temp[id])

            # 強化学習のポリシーをアップデート
            if i >= 1000:
                Agent[id].update()

        print(f"total power consumption: {building_state.power_balance:.2f}")
