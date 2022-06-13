from time import sleep
import math
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from simulator.bfs import BFSList
from simulator.io import BuildingAction
from rl import sac, new_sac
import argparse
import os
# とりあえず変数を定義
state_shape = (20,)  # エリアごとに順に1+5+5+5+1 + 3
action_shape = (4,)  # 各HVACの制御_ on or off (3つ) + Electric Storageの制御(1つ)
FIXED_TEMP = 28
enable_tensorboard = False

def write_to_tensorboard(bfs_list, state_obj, hvac_mode, mode, reward):
        
    writer.add_scalar(
        "temperature_area1", state_obj.areas[1].temperature, bfs_list[0].cur_steps)
    writer.add_scalar(
        "temperature_area2", state_obj.areas[2].temperature, bfs_list[0].cur_steps)
    writer.add_scalar(
        "temperature_area3", state_obj.areas[3].temperature, bfs_list[0].cur_steps)
    writer.add_scalar(
        "people_area1", state_obj.areas[1].people, bfs_list[0].cur_steps)

    writer.add_scalar(
        "hvac_mode", hvac_mode[1] , bfs_list[0].cur_steps)
    

    mode_dict = {
        'charge': 1,
        'stand_by': 0,
        'discharge': -1
    }
    # writer.add_scalar('charge_mode_per_price', price, mode_)
    writer.add_scalar('charge_mode_per_time', mode_dict[mode], bfs_list[0].cur_steps)
    writer.add_scalar('charge_ratio', state_obj.areas[4].facilities[0].charge_ratio, bfs_list[0].cur_steps)
    writer.add_scalar('reward', reward[0], bfs_list[0].cur_steps)

def cvt_state_to_ndarray(state,day, st):
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
    # 平日であれば [1,0] 祝日であれば[0,1]を追加 
    if day % 7 == 0 or day % 7 == 6:
        state_arr.extend([0,1])
    else:
        state_arr.extend([1,0])

    if st % 2 == 0:
        st *= 2* np.pi / (60*24)
        pe = np.sin(st)
    else:
        st *= 2* np.pi / (60*24)
        pe = np.cos(st)
    
    state_arr.append(pe)
    return np.array(state_arr)


def action_to_HVAC_mode(action):
    hvac_mode = np.where(action > 0, True,False)
    return hvac_mode


def action_to_ES(action):

    if - 1 <= action < -1 / 3:
        mode = 'charge'
    elif - 1 / 3 <= action <= 1 / 3:
        mode = 'stand_by'
    else:
        mode = 'discharge'
    return mode

def apply_fed_avg(Agent, N):

    # 各モデルを統一するパラメータを保存するためのzeoro-tensorを作成

    actor_model = {}
    critic_model = {}
    critic_target_model = {}
    actor_dict = Agent[0].actor.state_dict()
    critic_dict = Agent[0].critic.state_dict()
    critic_target_dict = Agent[0].critic_target.state_dict()
    with torch.no_grad():
        for u in actor_dict:
            actor_model[u] = torch.zeros(actor_dict[u].shape).to('cuda:0')
        
        for u in critic_dict:
            critic_model[u] = torch.zeros(critic_dict[u].shape).to('cuda:0')
            critic_target_model[u] = torch.zeros(critic_dict[u].shape).to('cuda:0')

        # 平均を求める
        for i in range(N):
            for u in Agent[i].actor.state_dict():
                actor_model[u].data.add_(Agent[i].actor.state_dict()[u].data.clone())
        
        for u in actor_model:
            actor_model[u].data.mul_(1 / N)
        

        for i in range(N):
            for u in Agent[i].critic.state_dict():
                critic_model[u].data.add_(Agent[i].critic.state_dict()[u].data.clone())
        
        for u in critic_model:
            critic_model[u].data.mul_(1 / N)
        
        for i in range(N):
            for u in Agent[i].critic_target.state_dict():
                critic_target_model[u].data.add_(Agent[i].critic_target.state_dict()[u].data.clone())
        
        for u in critic_target_model:
            critic_target_model[u].data.mul_(1 / N)
        
        # 各Agentにパラメータを分配
        for i in range(N):
            for u in Agent[i].actor.state_dict():
                Agent[i].actor.state_dict()[u].data.copy_(actor_model[u])
        for i in range(N):
            for u in Agent[i].critic.state_dict():
                Agent[i].critic.state_dict()[u].data.copy_(critic_model[u])
        
        for i in range(N):
            for u in Agent[i].critic_target.state_dict():
                Agent[i].critic_target.state_dict()[u].data.copy_(critic_target_model[u])
    print('update complete')

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    if enable_tensorboard:
        writer = SummaryWriter('setsuden')
    parser = argparse.ArgumentParser()
    parser.add_argument('--building_num', type=int, dest='N', default=1)
    parser.add_argument('--cuda_name', type=str, dest='cuda', default='cuda:0')
    parser.add_argument('--data_path', type=str, dest='data_path', default=None)
    parser.add_argument('--scenario', type=str, dest='scenario', default=None)
    
    args = parser.parse_args()
    N = args.N
    cuda = args.cuda
    data_path = args.data_path
    scenario = args.scenario
    bfs_list = BFSList(data_path)
    
    action = []    
    for i in range(N):
        action.append(BuildingAction())
        action[-1].add(area_id=1, facility_id=0, status=True, temperature=22)
        action[-1].add(area_id=2, facility_id=0, status=True, temperature=25)
        action[-1].add(area_id=3, facility_id=0, status=True, temperature=28)
        action[-1].add(area_id=4, facility_id=0, mode='stand_by')

    # 強化学習を行うエージェントを作成 (Soft-Actor-Critic という手法を仮に用いている)
    Agent = []
    for _ in range(N):
        Agent.append(sac.SAC(state_shape=state_shape,action_shape=action_shape, device='cuda:0' if torch.cuda.is_available() else 'cpu'))
    print(torch.cuda.is_available())    
    # ここはとりあえず状態, 行動, 報酬, 設定温度の変数を初期化
    state = np.zeros((N,*state_shape))
    next_state = np.zeros((N,*state_shape))
    action_ = np.zeros((N,*action_shape))
    reward = np.zeros((N,1))
    # hvac_mode = np.zeros((N,3))
    charge_ratio = 0
   
    for month in range(1):
        for day in range(31):
            # N: ビルの数
            if day <= 6:
                for i in range(N):
                    # 1日に1回全体のモデルを更新
                    for st in range(60 * 24):
                        if bfs_list[i].has_finished():
                            break
                        (state_obj, reward_obj) = bfs_list[i].step(action[i])

                        next_state[i] = cvt_state_to_ndarray(state_obj, day, st)
                        reward[i] = reward_obj.metric1

                        if bfs_list[i].cur_steps >= 1:
                            Agent[i].replay_buffer.add(state[i], action_[i], next_state[i], reward[i], done=False)
                        state[i] = next_state[i]

                        if bfs_list[i].cur_steps == 0:
                            continue

                        
                        action_[i], _ = Agent[i].choose_action(state[i])
                        
                        hvac_mode = action_to_HVAC_mode(action_[i][:-1])  # 一番最後のESは除く
                        # mode = action_to_ES(action_[i][-1])
                        # on/offの設定のみ
                        action[i].add(area_id=1, facility_id=0, status=hvac_mode[0], temperature = FIXED_TEMP)
                        action[i].add(area_id=2, facility_id=0, status=hvac_mode[1], temperature = FIXED_TEMP)
                        action[i].add(area_id=3, facility_id=0, status=hvac_mode[2], temperature = FIXED_TEMP)
                        action[i].add(area_id=4, facility_id=0, mode='stand_by')
                        
                        Agent[i].update()

                        if bfs_list[i].cur_steps % 60 == 0:
                            print(f"{i}-th building",end = "")
                            bfs_list[i].print_cur_state()
                            if i == 0:
                                if enable_tensorboard:
                                    write_to_tensorboard(bfs_list, state_obj, hvac_mode, mode, reward)
            
                # fedlated_learningを適用 (モデルのparameterを全体平均を用いて更新)
                apply_fed_avg(Agent,N)
                # new_sac.average_sac(Agent)
                print('completely update ')
        # save NN model
        #if month % 2 == 0:
        torch.save(Agent[0].actor.state_dict(), f'saved_model_{scenario}/actor_{month* 31}_num_{N}.pth')
        torch.save(Agent[0].critic.state_dict(), f'saved_model_{scenario}/critic_{month*31}_num_{N}.pth')
                            
    # save NN model
    torch.save(Agent[0].actor.state_dict(), f'saved_model_{scenario}/actor_num_{N}.pth')
    torch.save(Agent[0].critic.state_dict(), f'saved_model_{scenario}/critic_num_{N}.pth')