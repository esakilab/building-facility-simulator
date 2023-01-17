from evaluation.common import compare_elapsed_time_with_diffs, BASE_CONFIG_DICT


if __name__ == "__main__":
    config_pattern_diffs = [
        dict(steps_per_round=i) for i in [4, 8, 15, 60, 120, 240, 480]
    ]
    compare_elapsed_time_with_diffs(config_pattern_diffs, "change_round_steps")
