from time import sleep
import math
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from simulator.bfs import BFSList, BuildingFacilitySimulator
from rl import sac
from simulator.building import BuildingAction, BuildingState


def write_to_tensorboard(bfs: BuildingFacilitySimulator, action_arr: np.ndarray, reward: float):
    state = bfs.get_state()
    action = BuildingAction.from_ndarray(action_arr, bfs.areas)
        
    writer.add_scalar("reward", reward, bfs.cur_steps)

    writer.add_scalar("set_temperature_area1", action.areas[1].facilities[0].set_temperature, bfs.cur_steps)
    writer.add_scalar("set_temperature_area2", action.areas[2].facilities[0].set_temperature, bfs.cur_steps)
    writer.add_scalar("set_temperature_area3", action.areas[3].facilities[0].set_temperature, bfs.cur_steps)
    writer.add_scalar("temperature_area1", state.areas[1].temperature, bfs.cur_steps)
    writer.add_scalar("temperature_area2", state.areas[2].temperature, bfs.cur_steps)
    writer.add_scalar("temperature_area3", state.areas[3].temperature, bfs.cur_steps)
    
    mode_dict = {
        'charge': 1,
        'stand_by': 0,
        'discharge': -1
    }
    writer.add_scalar('charge_mode_per_time', mode_dict[action.areas[4].facilities[0].mode.value], bfs.cur_steps)
    writer.add_scalar('charge_ratio', state.areas[4].facilities[0].charge_ratio, bfs_list[0].cur_steps)


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
    reward += - LAMBDA2 * (np.where((T_MIN - area_temp) < 0, 0, (T_MIN - area_temp)).sum())
    reward += - LAMBDA2 * (np.where((area_temp - T_MAX) < 0, 0, (area_temp - T_MAX)).sum())
    reward += - LAMBDA3 * state.electric_price_unit * state.power_balance
    reward += LAMBDA4 * state.areas[4].facilities[0].charge_ratio

    return np.array([reward])


if __name__ == "__main__":
    writer = SummaryWriter(log_dir="./logs/3federated_only_area1")
    bfs_list = BFSList(calc_reward, './input_xmls', 3)

    Agents = [sac.SAC(state_shape=bfs_list[i].get_state_shape(),
                      action_shape=bfs_list[i].get_action_shape(), 
                      device='cpu') for i in range(len(bfs_list))]

    states = [np.zeros(*bfs_list[i].get_state_shape()) for i in range(len(bfs_list))]
    actions = [np.zeros(*bfs_list[i].get_action_shape()) for i in range(len(bfs_list))]

    reward = np.zeros(1)
    temp = np.zeros(3)
    charge_ratio = 0


    while True:
        for _ in range(60 * 24):
            # print(Agents[0].critic.output1.weight[0, :3].data, Agents[1].critic.output1.weight[0, :3].data)

            for i in range(len(bfs_list)):
                if bfs_list[i].has_finished():
                    continue
            
                (next_state, reward) = bfs_list[i].step(actions[i])

                if bfs_list[i].cur_steps >= 1:
                    Agents[i].replay_buffer.add(
                        states[i], actions[i], next_state, reward[0], done=False)
                states[i] = next_state

                if bfs_list[i].cur_steps == 0:
                    continue

                if bfs_list[i].cur_steps >= 100:
                    actions[i], _ = Agents[i].choose_action(states[i])
                else:
                    actions[i] = np.random.uniform(low=-1, high=1, size=bfs_list[i].get_action_shape()[0])

                Agents[i].update()

                if i == 0:
                    write_to_tensorboard(bfs_list[i], actions[i], reward[0])
                    
                    if bfs_list[i].cur_steps % 60 == 0:
                        bfs_list[i].print_cur_state()
            
        sac.average_sac(Agents)

        print("merged models!")
