from distributed_platform.experiment import Experiment, ExperimentConfig
from main import calc_reward
from rl.sac import SAC, average_sac


if __name__ == "__main__":
    config = ExperimentConfig.parse_file("./data/json/example/experiment_config_small.json")
    calc_reward_dict = dict(group15=calc_reward, group20=calc_reward, group25=calc_reward)
    
    exp = Experiment(SAC, average_sac, calc_reward_dict, config)
    exp.run()
