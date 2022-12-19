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
    "ssh://10.17.104.170", "ssh://10.17.108.100", "ssh://10.17.108.101", "ssh://10.17.108.102", "ssh://10.17.108.111", "ssh://10.17.108.137", "ssh://10.17.108.226", "ssh://10.17.108.82",
    "ssh://10.17.108.83",  "ssh://10.17.108.84",  "ssh://10.17.108.85",  "ssh://10.17.108.86",  "ssh://10.17.108.87",  "ssh://10.17.108.90",  "ssh://10.17.108.91",  "ssh://10.17.108.92",
    "ssh://10.17.108.95",  "ssh://10.17.108.96",  "ssh://10.17.108.97",  "ssh://10.17.108.98",  "ssh://10.17.108.99",  "ssh://10.17.109.159", "ssh://10.17.109.247", "ssh://10.17.110.12",
    "ssh://10.17.110.133", "ssh://10.17.110.134", "ssh://10.17.110.226", "ssh://10.17.110.227", "ssh://10.17.110.75",  "ssh://10.17.110.87",  "ssh://10.17.111.161", "ssh://10.17.111.86",
]

BASE_CONFIG_DICT = dict(
    global_node_ip="10.17.104.157",
    selection_port=11113,
    reporting_port=11114,
    local_node_urls=LOCAL_NODE_URLS, 
    total_client_num=4, 
    round_client_num=4, 
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
    all_diff_fields = functools.reduce(operator.or_, (diff.keys() for diff in config_diffs), set())
    print(all_diff_fields)

    log_columns = list(all_diff_fields) + ["elapsed_time"]
    log_file_path = LOG_DIR / f"{experiment_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.tsv"

    with log_file_path.open('w') as f:
        f.write(list_to_row(log_columns))

    for diff in config_diffs:
        exp_dict = BASE_CONFIG_DICT | diff

        exp_dict["elapsed_time"] = Experiment(
            SAC, 
            average_sac, 
            CALC_REWARD_DICT, 
            ExperimentConfig.parse_obj(exp_dict)
        ).run()


        with log_file_path.open('a') as f:
            row_values = [str(exp_dict[key]) for key in log_columns]
            f.write(list_to_row(row_values))
