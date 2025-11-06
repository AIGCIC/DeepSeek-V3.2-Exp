# To do this test, you need to manually set the layer number of DSAShowHands to be same as the number in config.
import pytest
import torch
import re
import json
from model import ModelArgs
from tilert.ops import tilert_init
from tilert.models.deepseek_v3_2.model import (
    Transformer as TilertDeepSeekV32Transformer,
    MLA as TilertMLA,
    RMSNormMLP as TilertMLP,
    RMSNormMoE as TilertMoE
)
from tilert.models.deepseek_v3_2.modules.dsa_show_hands import ShowHandsDSALayer
from tilert.models.deepseek_v3_2.modules.decode_layer import MlaParams, MLPParams, MoEParams
from tilert.models.deepseek_v3_2.modules.dsa_show_hands import LLMHeadParams
from func_test import convert_state_dict
# bf16 for better alignment test
from model_bf16 import Transformer as DeepSeekV32Transformer

@pytest.fixture(autouse=True)
def setup():
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(1234)

def init_show_hands_mla_params(mla_params: list[torch.Tensor], mla : TilertMLA):
    x_rmsnorm_gamma = mla_params[0]
    qkv_wa_weights = mla_params[1]
    qkv_wa_scales = mla_params[2]
    k_weights = mla_params[3]
    k_bias = mla_params[4]
    q_rmsnorm_gamma = mla_params[5]
    q_wb_weights = mla_params[6]
    q_wb_scales = mla_params[7]
    id_score_weights = mla_params[8]
    wkv_b1_weights = mla_params[9]
    wkv_b1_scales = mla_params[10]
    kv_rmsnorm_gamma = mla_params[11]
    wkv_b2_weights = mla_params[12]
    wkv_b2_scales = mla_params[13]
    unproj_weights = mla_params[14]
    unproj_scales = mla_params[15]
    x_rmsnorm_gamma.copy_(mla.rmsnorm_proj_qkvwa_ki_rope.tilert_attn_norm_weight)

    qkv_wa_weights.copy_(mla.rmsnorm_proj_qkvwa_ki_rope.tilert_wqkv_a)
    qkv_wa_scales.copy_(mla.rmsnorm_proj_qkvwa_ki_rope.tilert_wqkv_a_scales)
    k_weights.copy_(mla.layernorm_rope_rotate.k_norm.weight)
    k_bias.copy_(mla.layernorm_rope_rotate.k_norm.bias)
    q_rmsnorm_gamma.copy_(mla.rmsnorm_proj_qwb_iq_rope.tilert_q_norm_weight)
    q_wb_weights.copy_(mla.rmsnorm_proj_qwb_iq_rope.tilert_wq_b_full)
    q_wb_scales.copy_(mla.rmsnorm_proj_qwb_iq_rope.tilert_wq_b_full_scales)
    id_score_weights.copy_(mla.proj_w.tilert_weights_proj)
    wkv_b1_weights.copy_(mla.proj_qwb.tilert_wkv_b_a)
    wkv_b1_scales.copy_(mla.proj_qwb.tilert_wkv_b_a_scales)
    kv_rmsnorm_gamma.copy_(mla.kv_rmsnorm.tilert_kv_norm_weight)
    wkv_b2_weights.copy_(mla.proj_qwb.tilert_wkv_b_b)
    wkv_b2_scales.copy_(mla.proj_qwb.tilert_wkv_b_b_scales)
    #with torch.no_grad():
    #    mla.unproj_o_allreduce.wo.weight.copy_(torch.ones_like(mla.unproj_o_allreduce.wo.weight, dtype=torch.torch.float32).to(mla.unproj_o_allreduce.wo.weight.dtype))
    unproj_weights.copy_(mla.unproj_o_allreduce.wo.weight)
    #with torch.no_grad():
    #    mla.unproj_o_allreduce.wo.scale.copy_(torch.ones_like(mla.unproj_o_allreduce.wo.scale))
    mat_scale = mla.unproj_o_allreduce.wo.scale.reshape((56, 1, 16)).repeat(1, 16, 1).reshape(896, 16)
    unproj_scales.copy_(mat_scale)

def init_show_hands_mlp_params(mlp_params: list[torch.Tensor], mlp : TilertMLP):
    unproj_o_gamma = mlp_params[0]
    upgate_weights = mlp_params[1]
    upgate_scales = mlp_params[2]
    down_weights = mlp_params[3]
    down_scales = mlp_params[4]
    unproj_o_gamma.copy_(mlp.norm_up_gate.tilert_rms_norm_weight.cpu())
    upgate_weights.copy_(mlp.norm_up_gate.tilert_weights.cpu())
    upgate_scales.copy_(mlp.norm_up_gate.tilert_scales.cpu())
    weight = mlp.down.w2.weight.reshape(7168, 9, 256).transpose(0, 1).cpu()
    down_weights.copy_(weight)
    scale = mlp.down.w2.scale.reshape(56, 9, 2).transpose(0, 1).cpu()
    scale = scale.reshape(9, 56, 1, 2).repeat(1, 1, 16, 1).reshape(9, 128, 14).cpu()
    padding_zeros = torch.zeros(9, 128, 2).cpu()#.to(scale.device)
    mat_scale_tilert = torch.cat([scale, padding_zeros], dim=2)
    mat_scale_tilert = mat_scale_tilert.reshape(9, 1024, 2).cpu()
    down_scales.copy_(mat_scale_tilert)

def init_show_hands_moe_params(moe_params: list[torch.Tensor], moe : TilertMoE):
    unproj_o_gamma = moe_params[0]
    exp_proj_weights = moe_params[1]
    exp_bias = moe_params[2]
    exp_upgate_weights = moe_params[3]
    exp_upgate_scales = moe_params[4]
    exp_down_weights = moe_params[5]
    exp_down_scales = moe_params[6]
    unproj_o_gamma.copy_(moe.rmsnorm_expert.tilert_rms_norm_weight.cpu())
    exp_proj_weights.copy_(moe.rmsnorm_expert.tilert_proj_weight.cpu())
    exp_bias.copy_(moe.routed_up_gate_silu.tilert_route_gate_bias.cpu())
    exp_upgate_weights.copy_(moe.routed_up_gate_silu.tilert_experts_weight.cpu())
    exp_upgate_scales.copy_(moe.routed_up_gate_silu.tilert_experts_scales)
    down_weights = torch.cat((moe.down.shared_experts_w2.weight.unsqueeze(0),
                              *[w.weight.unsqueeze(0) for w in moe.down.experts_w2]), dim=0)
    exp_down_weights.copy_(down_weights)
    down_scales = torch.cat((moe.down.shared_experts_w2.scale.unsqueeze(0),
                              *[w.scale.unsqueeze(0) for w in moe.down.experts_w2]), dim=0)
    down_scales = (
                down_scales.reshape(257, 56, 1, 2).repeat(1, 1, 16, 1).reshape(257, 128, 14)
            )
    padding_zeros = torch.zeros((257, 128, 2), dtype=down_scales.dtype, device=down_scales.device)
    down_scales = torch.cat([down_scales, padding_zeros], dim=2)
    down_scales = down_scales.reshape(257, 1024, 2)
    exp_down_scales.copy_(down_scales)

def init_show_hands_head_params(head_params: list[torch.Tensor],
                                model : TilertDeepSeekV32Transformer):
    hidden_rms_gamma = head_params[0]
    head_proj_weights = head_params[1]
    hidden_rms_gamma.copy_(model.norm.weight.cpu())
    head_proj_weights.copy_(model.head.weight.cpu())

def zero_show_hands_mla_pre_allreduce_params(mla_params: list[torch.Tensor]):
    unproj_weights = mla_params[14]
    unproj_scales = mla_params[15] # unproj_allreduce
    unproj_weights.copy_(torch.zeros_like(unproj_weights, dtype=torch.float32).to(unproj_weights.dtype))
    unproj_scales.copy_(torch.zeros((896, 16))) # set to zero

def zero_show_hands_mlp_pre_allreduce_params(mlp_params: list[torch.Tensor]):
    down_weights = mlp_params[3]
    down_weights.copy_(torch.zeros_like(down_weights, dtype=torch.float32).to(down_weights.dtype))
    down_scales = mlp_params[4]
    down_scales.copy_(torch.zeros_like(down_scales))

def zero_show_hands_moe_pre_allreduce_params(moe_params: list[torch.Tensor]):
    exp_down_weights = moe_params[5]
    exp_down_weights.copy_(torch.zeros_like(exp_down_weights, dtype=torch.float32).to(exp_down_weights.dtype))
    exp_down_scales = moe_params[6]
    exp_down_scales.copy_(torch.zeros_like(exp_down_scales))

@pytest.mark.parametrize("model_config", [])
def test_e2e_forward_pass(model_config):
    tilert_init()

    with open(model_config, "r", encoding="utf-8") as f_json:
        model_config = json.load(f_json)
    model_args = ModelArgs(**model_config)

    input_token = torch.randint(0, model_args.vocab_size, (1, 1))
    start_pos = 127
    #hidden = torch.randn(1, 1, 7168)

    tilert_model = TilertDeepSeekV32Transformer(model_args)
    tilert_model.enable_tilert(True)

    origin_model = DeepSeekV32Transformer(model_args)
    conv_dict = convert_state_dict(tilert_model.state_dict())
    origin_model.load_state_dict(conv_dict)

    dsa_show_hands = ShowHandsDSALayer(model_args.max_seq_len)
    dsa_show_hands.init()

    mla_params = MlaParams.num_params()
    mlp_params = MLPParams.num_params()
    moe_params = MoEParams.num_params()
    # We only use device 0's result, since TilertDeepSeekV32Transformer only supports 1 device
    with torch.cuda.device(0):
        _, params, _ = dsa_show_hands.multi_devices_results[0]
        cur_offset = 0
        # You need to make sure that DSAShowHands has same number of layers !!!
        for i in range(model_args.n_dense_layers):
            init_show_hands_mla_params(params[cur_offset:], tilert_model.layers[i].attn)
            init_show_hands_mlp_params(params[cur_offset + mla_params:], tilert_model.layers[i].ffn)
            cur_offset += mla_params + mlp_params
        for i in range(model_args.n_dense_layers, model_args.n_layers):
            init_show_hands_mla_params(params[cur_offset:], tilert_model.layers[i].attn)
            init_show_hands_moe_params(params[cur_offset + mla_params:], tilert_model.layers[i].ffn)
            cur_offset += mla_params + moe_params
        init_show_hands_head_params(params[cur_offset:], tilert_model)

    # Set other devices' pre-allreduce weights to zero
    for i in range(1, 8):
        with torch.cuda.device(i):
            _, params, _ = dsa_show_hands.multi_devices_results[i]
            cur_offset = 0
            # You need to make sure that DSAShowHands has same number of layers !!!
            for i in range(model_args.n_dense_layers):
                zero_show_hands_mla_pre_allreduce_params(params[cur_offset:])
                zero_show_hands_mlp_pre_allreduce_params(params[cur_offset + mla_params:])
                cur_offset += mla_params + mlp_params
            for i in range(model_args.n_dense_layers, model_args.n_layers):
                zero_show_hands_mla_pre_allreduce_params(params[cur_offset:])
                zero_show_hands_moe_pre_allreduce_params(params[cur_offset + mla_params:])
                cur_offset += mla_params + moe_params

    with torch.no_grad():
        hidden = tilert_model.embed(input_token)
        freq_cis = tilert_model.freqs_cis[start_pos : start_pos + 1]
        # Use golden
        tilert_model.enable_tilert(False)
        logits = tilert_model(input_token, start_pos)
        # Original model
        logits_origin = origin_model(input_token, start_pos)

    """
    res_attn = tilert_model.layers[0].attn.forward(hidden, start_pos, freq_cis, None)
    # add residual
    res_with_residual = (res_attn.cpu().float() + hidden.cpu().float()).to(torch.bfloat16).cuda()
    # mlp
    res_ffn = tilert_model.layers[0].ffn.forward(res_with_residual)
    # add residual
    res = (res_ffn.cpu().float() + res_with_residual.cpu().float()).to(torch.bfloat16).cuda()

    res_attn = tilert_model.layers[1].attn.forward(res, start_pos, freq_cis, None)
    # add residual
    res_with_residual = (res_attn.cpu().float() + hidden.cpu().float()).to(torch.bfloat16).cuda()
    # moe
    res_ffn, indices, all_out = tilert_model.layers[1].ffn.forward(res_with_residual)
    all_out = torch.cat([all_out[-1:], all_out[:-1]], dim=0)
    # add residual
    print(res_with_residual)
    res = (res_ffn.cpu().float() + res_with_residual.cpu().float()).to(torch.bfloat16).cuda()
    """

    
    exec_stream = torch.cuda.Stream()
    with torch.cuda.stream(exec_stream):
        res_show_hands = dsa_show_hands.tilert_forward_mt(
            hidden, torch.tensor(start_pos), torch.view_as_real(freq_cis).reshape(1, 64), None)
    q = res_show_hands[0][0][0]
    q_nope = res_show_hands[0][0][12]
    q_pe = res_show_hands[0][0][6]
    x_rmsnorm = res_show_hands[0][0][27]
    exp_out = res_show_hands[0][0][26]
    proj_o = res_show_hands[0][0][19]
    unproj_o = res_show_hands[0][0][20]
    scores = res_show_hands[0][0][21]
    x_mlp_in = res_show_hands[0][0][22]
    up_gate = res_show_hands[0][0][23]
    sel_probs = res_show_hands[0][0][24]
    sel_indices = res_show_hands[0][0][25]
    logits_out = res_show_hands[0][0][31]
    print('ShowHands indices: ', sel_indices.sort()[0])

    gt = logits.cpu().flatten().float()
    sh = logits_out.cpu().flatten().float()
    gt_origin = logits_origin.cpu().flatten().float()

    relative_error = torch.norm(gt - sh) / torch.norm(gt)
    cos_similarity = torch.nn.functional.cosine_similarity(gt, sh, dim=-1)
    # cosine similarity
    print('Relative error: ', relative_error.cpu().item())
    print('Cosine: ', cos_similarity.cpu().item())

    relative_error = torch.norm(gt_origin - sh) / torch.norm(gt_origin)
    cos_similarity = torch.nn.functional.cosine_similarity(gt_origin, sh, dim=-1)
    # cosine similarity
    print('Relative error(Origin): ', relative_error.cpu().item())
    print('Cosine(Origin): ', cos_similarity.cpu().item())

def main():
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(1234)
    test_e2e_forward_pass("config_671B_v3.2_layer2_device1.json")


if __name__ == "__main__":
    main()
