"""
This script is used to summarize the external timing of the model.
"""

import os
import json

if __name__ == "__main__":
    log_dir_ : str = './profile_logs'

    v3_2_model_mlp_layers = [
        'RMSNorm_ProjQ_WQKVIa_RoPE',
        'LayerNormRoPERotate',
        'RMSNorm_ProjQ_WQIb_RoPE',
        'Rotate',
        'ProjQ_WIb',
        'Indexer',
        'TopK',
        'ProjQ_WKBb',
        'RMSNorm_KV',
        'Sparse_FlashMLA',
        'UnProjO_WKVb',
        'UnprojOAllreduce',
        'RMSNorm_MLP_Up_Gate_SiLU',
        'MLPDown'
        ]

    v3_2_model_moe_layers = [
        'RMSNorm_ProjQ_WQKVIa_RoPE',
        'LayerNormRoPERotate',
        'RMSNorm_ProjQ_WQIb_RoPE',
        'Rotate',
        'ProjQ_WIb',
        'Indexer',
        'TopK',
        'ProjQ_WKBb',
        'RMSNorm_KV',
        'Sparse_FlashMLA',
        'UnProjO_WKVb',
        'UnprojOAllreduce',
        'RMSNorm_Expert_Proj',
        'Expert_Select_Up_Gate_SiLU',
        'MoEDown'
        ]

    def sum_timing(layers, log_dir, summary_dict, timing_key):
        tot_time = 0.0
        for layer in layers:
            json_file = os.path.join(log_dir, f'{layer}.json')
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f_:
                    data = json.load(f_)
                summary_dict[layer] = data[timing_key]
            else:
                summary_dict[layer] = 0.0
            tot_time += summary_dict[layer]
        summary_dict['Total'] = tot_time

    summary_data = {}
    summary_data['mlp_cpu'] = {}
    summary_data['moe_cpu'] = {}
    summary_data['mlp_gpu'] = {}
    summary_data['moe_gpu'] = {}

    sum_timing(
        v3_2_model_mlp_layers, log_dir_, summary_data['mlp_cpu'], 'cpu_time_in_us')
    sum_timing(
        v3_2_model_mlp_layers, log_dir_, summary_data['mlp_gpu'], 'gpu_time_in_us')
    sum_timing(
        v3_2_model_moe_layers, log_dir_, summary_data['moe_cpu'], 'cpu_time_in_us')
    sum_timing(
        v3_2_model_moe_layers, log_dir_, summary_data['moe_gpu'], 'gpu_time_in_us')

    tot_time = summary_data['moe_cpu']['Total'] * 58 + summary_data['mlp_cpu']['Total'] * 3
    summary_data['token_time_in_us'] = tot_time
    summary_data['tokens_per_sec'] = 1000000.0 / summary_data['token_time_in_us']

    with open(os.path.join(log_dir_, 'summary_data.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4)
