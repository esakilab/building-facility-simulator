from evaluation.common import compare_elapsed_time_with_diffs


if __name__ == "__main__":

    config_pattern_diffs = [
        dict(total_client_num=2 ** i) for i in range(2, 15)
    ]
    compare_elapsed_time_with_diffs(config_pattern_diffs, "change_client_num")
