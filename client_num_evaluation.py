from evaluation.common import compare_elapsed_time_with_diffs


if __name__ == "__main__":

    config_pattern_diffs = [
        dict(total_client_num=2 ** i) for i in range(2, 15)
    ]
    compare_elapsed_time_with_diffs(config_pattern_diffs, "change_client_num")
    """
    ↑の結果
    シミュレーションするビル数が増えているので、ビルが増えると遅くなるのは当然
    ただし、32までは1マシン一ビル以下なので、同じにできそうさはある
    同じにならないのは、selectされていないclientのシミュレーションをselectされたときにまとめて行うからっぽいかも
    [
        ({'total_client_num': 4}, 282.3401296492666), 
        ({'total_client_num': 8}, 331.9261703705415), 
        ({'total_client_num': 16}, 442.9217373272404), 
        ({'total_client_num': 32}, 648.2622807007283), 
        ({'total_client_num': 64}, 1053.715103359893), 
        ({'total_client_num': 128}, 1790.9957465594634), 
        ({'total_client_num': 256}, 3061.558152858168), 
        ({'total_client_num': 512}, 4590.699866762385), 
        ({'total_client_num': 1024}, 4917.746338367462)
    ]
    """
