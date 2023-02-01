from datetime import datetime
import functools
import operator
from pathlib import Path
from typing import Any
from main import calc_reward
from distributed_platform.experiment import Experiment, ExperimentConfig
from main import calc_reward
from rl.sac import SAC, average_sac

LOCAL_NODE_URLS = [
    "ssh://10.17.108.124", "ssh://10.17.108.145", "ssh://10.17.108.193", "ssh://10.17.109.117",
    "ssh://10.17.109.123", "ssh://10.17.109.125", "ssh://10.17.109.133", "ssh://10.17.109.143",
    "ssh://10.17.109.145", "ssh://10.17.109.161", "ssh://10.17.109.172", "ssh://10.17.109.192",
    "ssh://10.17.109.207", "ssh://10.17.109.63",  "ssh://10.17.109.93",  "ssh://10.17.110.106",
    "ssh://10.17.110.112", "ssh://10.17.110.167", "ssh://10.17.110.202", "ssh://10.17.110.214",
    "ssh://10.17.110.224", "ssh://10.17.110.241", "ssh://10.17.110.45",  "ssh://10.17.110.58",
    "ssh://10.17.110.95",  "ssh://10.17.111.0",   "ssh://10.17.111.132", "ssh://10.17.111.171",
    "ssh://10.17.111.42",  "ssh://10.17.111.55",  "ssh://10.17.111.83",  "ssh://10.17.111.85",
]

BASE_CONFIG_DICT = dict(
    global_node_ip="10.17.111.33",
    selection_port=11113,
    reporting_port=11114,
    local_node_urls=LOCAL_NODE_URLS, 
    total_client_num=256, 
    round_client_num=32, 
    start_time="2020-08-01T00:00:00",
    steps_per_round=60,
    total_steps=10080,
    model_tag_to_config_dir_path={"onlygroup": "data/json"}
)

CALC_REWARD_DICT = dict(onlygroup=calc_reward)

LOG_DIR = Path("evaluation/log")

def list_to_row(l: list[str]) -> str:
    return '\t'.join(l) + '\n'

def compare_elapsed_time_with_diffs(config_diffs: list[dict[str, Any]], experiment_name: str):
    all_diff_fields: set = functools.reduce(operator.or_, (diff.keys() for diff in config_diffs), set())

    if "local_node_urls" in all_diff_fields:
        all_diff_fields.remove("local_node_urls")
        all_diff_fields.add("local_node_num")
    
    print(all_diff_fields)

    log_columns = list(all_diff_fields) + ["elapsed_time"]
    log_file_path = LOG_DIR / f"{experiment_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.tsv"

    with log_file_path.open('w') as f:
        f.write(list_to_row(log_columns))

    for diff in config_diffs:
        exp_dict = BASE_CONFIG_DICT | diff

        layer_num = exp_dict.pop("model_layer_num", None)
        if layer_num:
            SAC.LAYER_NUM = layer_num
        else:
            SAC.LAYER_NUM = 1

        exp_dict["elapsed_time"] = Experiment(
            SAC, 
            average_sac, 
            CALC_REWARD_DICT, 
            ExperimentConfig.parse_obj(exp_dict)
        ).run()

        if "local_node_urls" in exp_dict:
            url_list = exp_dict.pop("local_node_urls")
            exp_dict["local_node_num"] = len(url_list)

        if layer_num:
            exp_dict.update(model_layer_num=layer_num)

        with log_file_path.open('a') as f:
            row_values = [str(exp_dict[key]) for key in log_columns]
            f.write(list_to_row(row_values))
