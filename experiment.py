"""
Usage: python3 experiment.py ([exp_name]=debug_{datetime})
"""

from time import sleep
from sys import argv
import datetime
import os
import numpy as np
from collections import defaultdict

from simulator.bfs import BuildingFacilitySimulator
from simulator.io import BuildingAction

START_DT = datetime.datetime(year=2020, month=8, day=10)

def step_to_datetime(step):
    return START_DT + datetime.timedelta(minutes=step)

def normal_action_func(_):
    return True, 'stand_by'

def saving_action_func(bfs):
    return bfs.get_next_area_env(1).people > 0, 'stand_by'

def storing_action_func(bfs):
    hvac_status = bfs.get_next_area_env(1).people > 0
    es_mode = 'discharge' if 10 <= step_to_datetime(bfs.cur_steps).hour <= 18 else 'charge'
    return hvac_status, es_mode

exp_modes = ['normal', 'saving', 'storing']
action_func_dict = {
    'normal': normal_action_func,
    'saving': saving_action_func,
    'storing': storing_action_func
}

def run_experiment(log_path):
    if os.path.exists(log_path):
        print(f"{log_path} already exists! Abotring.")
        exit(1)

    bfs_dict = defaultdict(lambda: BuildingFacilitySimulator('./input_xmls/example/BFS_environment.xml'))
    total_charge_dict = defaultdict(int)

    data_arr = []

    for i in range(60 * 24 * 7):
        data_row = [i, bfs_dict['normal'].get_next_ext_env().temperature]
        _actions = []

        for exp_mode in exp_modes:
            hvac_status, es_mode = action_func_dict[exp_mode](bfs_dict[exp_mode])
            # print(i, hvac_status, es_mode)
            action = BuildingAction()
            action.add(area_id=1, facility_id=0, status=hvac_status, temperature=25)
            action.add(area_id=2, facility_id=0, mode=es_mode)
                
            (state_obj, _) = bfs_dict[exp_mode].step(action)

            total_charge_dict[exp_mode] += state_obj.power_balance * state_obj.electric_price_unit
            data_row.extend([state_obj.areas[1].temperature, state_obj.power_balance, total_charge_dict[exp_mode]])
            _actions.extend([hvac_status, es_mode])

        data_arr.append(data_row)
        # print(f"{step_to_datetime(i)}: {_actions}")

    np.savetxt(
        log_path,
        data_arr,
        fmt=['%d', '%.2f'] + ['%.2f', '%.2f', '%.2f'] * len(exp_modes),
        delimiter='\t',
        header='\t'.join(
            sum(
                ([f'{mode}_room_temp', f'{mode}_power', f'{mode}_total_charge'] for mode in exp_modes), 
                start=['step', 'ext_temp']
            )
        ),
        comments=''
    )
    print(f'saved to {log_path}')


if __name__ == "__main__":
    exp_name = argv[1] if len(argv) > 1 else f'debug_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'

    print(f'starting with exp_name: {exp_name}')
    run_experiment(f'./logs/{exp_name}.tsv')

    
