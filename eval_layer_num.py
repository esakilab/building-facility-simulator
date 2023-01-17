from evaluation.common import compare_elapsed_time_with_diffs, BASE_CONFIG_DICT


if __name__ == "__main__":
    BASE_CONFIG_DICT.update(
        round_client_num=32,
    )
    config_pattern_diffs = [
        dict(model_layer_num=i) for i in [1, 2, 3, 4, 5]
    ]
    compare_elapsed_time_with_diffs(config_pattern_diffs, "change_sac_layer_num")
