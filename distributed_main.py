from distributed_platform.experiment import Experiment, ExperimentConfig
from main import calc_reward
from rl.sac import SAC, average_sac


if __name__ == "__main__":
    config = ExperimentConfig.parse_file("./data/json/example/experiment_config_small.json")
    exp = Experiment(SAC, average_sac, calc_reward, config)
    exp.run()
