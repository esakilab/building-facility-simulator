from evaluation.common import compare_elapsed_time_with_diffs, BASE_CONFIG_DICT, LOCAL_NODE_URLS


if __name__ == "__main__":
    BASE_CONFIG_DICT.update(
        total_client_num=128,
        round_client_num=16,
    )
    config_pattern_diffs = [
        dict(local_node_urls=LOCAL_NODE_URLS[:i]) for i in [32, 16, 8, 4, 2, 1]
    ]
    compare_elapsed_time_with_diffs(config_pattern_diffs, "change_node_num")
