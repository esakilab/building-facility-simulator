from scalability_experiment import experiment


if __name__ == "__main__":
    experiment(
        total_clients=8, 
        nodes=8, 
        total_steps=60 * 24 * (31 * 12 + 1), # 12ヶ月 + 1日分
        round_client_num=8, 
        log_states=True,
        cycle_env_iter=True)
