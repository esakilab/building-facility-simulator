from evaluation.common import compare_elapsed_time_with_diffs, BASE_CONFIG_DICT


if __name__ == "__main__":
    BASE_CONFIG_DICT.update(
        total_client_num=256,
    )

    config_pattern_diffs = [
        dict(round_client_num=i) for i in [1, 2, 4, 8, 16, 32, 64, 128, 256]
    ]
    compare_elapsed_time_with_diffs(config_pattern_diffs, "change_round_client_num")
