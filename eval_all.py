from evaluation.common import LOCAL_NODE_URLS, compare_elapsed_time_with_diffs


if __name__ == "__main__":

    client_num_diffs = [
        dict(total_client_num=i) for i in [32, 64, 128, 256, 512, 1024, 2048, 4096]
    ]
    compare_elapsed_time_with_diffs(client_num_diffs, "change_client_num")


    layer_num_diffs = [
        dict(model_layer_num=i) for i in [1, 2, 3, 4, 5]
    ]
    compare_elapsed_time_with_diffs(layer_num_diffs, "change_sac_layer_num")

    node_num_diffs = [
        dict(local_node_urls=LOCAL_NODE_URLS[:i]) for i in [32, 16, 8, 4, 2, 1]
    ]
    compare_elapsed_time_with_diffs(node_num_diffs, "change_node_num")

    round_cli_num_diffs = [
        dict(round_client_num=i) for i in [1, 2, 4, 8, 16, 32, 64, 128, 256]
    ]
    compare_elapsed_time_with_diffs(round_cli_num_diffs, "change_round_client_num")

    round_step_num_diffs = [
        dict(steps_per_round=i) for i in [4, 8, 15, 30, 60, 120, 240, 480]
    ]
    compare_elapsed_time_with_diffs(round_step_num_diffs, "change_round_steps")
