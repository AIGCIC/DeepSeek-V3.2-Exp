import pytest
import torch
import re

import json
from model import ModelArgs

from model_bf16 import Transformer as DeepSeekV32Transformer
from tilert.models.deepseek_v3_2.model import (
    Transformer as TilertDeepSeekV32Transformer,
)


@pytest.fixture(autouse=True)
def setup():
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)

def convert_state_dict(tilert_state_dict: dict) -> dict:
    """
    Convert the state dict of the Tilert DeepSeekV3 model to the state dict of the DeepSeekV3 model.
    """
    key_casting_maps = {
                r"layers\.(\d+)\.ffn\.norm_up_gate\.rms_norm.weight": r"layers.\1.ffn_norm.weight",  # noqa: E501
        # r"layers\.(\d+)\.ffn\.route_proj\.norm_weight": r"layers.\1.ffn_norm.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.rmsnorm_expert\.rms_norm.weight": r"layers.\1.ffn_norm.weight",  # noqa: E501
        # MLP weight
        r"layers\.(\d+)\.ffn\.norm_up_gate.w1.weight": r"layers.\1.ffn.w1.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.down.w2.weight": r"layers.\1.ffn.w2.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.norm_up_gate.w3.weight": r"layers.\1.ffn.w3.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.norm_up_gate.w1.scale": r"layers.\1.ffn.w1.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.down.w2.scale": r"layers.\1.ffn.w2.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.norm_up_gate.w3.scale": r"layers.\1.ffn.w3.scale",  # noqa: E501
        # Project weight
        r"layers\.(\d+)\.ffn\.rmsnorm_expert\.proj_weight": r"layers.\1.ffn.gate.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.route_gate\.bias": r"layers.\1.ffn.gate.bias",  # noqa: E501
        # Expert weight
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.experts_w1\.(\d+)\.weight": r"layers.\1.ffn.experts.\2.w1.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.experts_w1\.(\d+)\.scale": r"layers.\1.ffn.experts.\2.w1.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.down.experts_w2\.(\d+)\.weight": r"layers.\1.ffn.experts.\2.w2.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.down.experts_w2\.(\d+)\.scale": r"layers.\1.ffn.experts.\2.w2.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.experts_w3\.(\d+)\.weight": r"layers.\1.ffn.experts.\2.w3.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.experts_w3\.(\d+)\.scale": r"layers.\1.ffn.experts.\2.w3.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.shared_experts_w1.weight": r"layers.\1.ffn.shared_experts.w1.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.shared_experts_w1.scale": r"layers.\1.ffn.shared_experts.w1.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.down.shared_experts_w2.weight": r"layers.\1.ffn.shared_experts.w2.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.down.shared_experts_w2.scale": r"layers.\1.ffn.shared_experts.w2.scale",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.shared_experts_w3.weight": r"layers.\1.ffn.shared_experts.w3.weight",  # noqa: E501
        r"layers\.(\d+)\.ffn\.routed_up_gate_silu\.up_gate_silu\.shared_experts_w3.scale": r"layers.\1.ffn.shared_experts.w3.scale",  # noqa: E501
        
        # norm attn
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qkvwa_ki_rope\.wq_a\.weight": r"layers.\1.attn.wq_a.weight",
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qkvwa_ki_rope\.wq_a\.scale": r"layers.\1.attn.wq_a.scale", 
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qkvwa_ki_rope\.wkv_a\.weight": r"layers.\1.attn.wkv_a.weight",
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qkvwa_ki_rope\.wkv_a\.scale": r"layers.\1.attn.wkv_a.scale",
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qkvwa_ki_rope\.attn_norm\.weight": r"layers.\1.attn_norm.weight",
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qkvwa_ki_rope\.wk\.weight": r"layers.\1.attn.indexer.wk.weight",
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qkvwa_ki_rope\.wk\.scale": r"layers.\1.attn.indexer.wk.scale",

        r"layers\.(\d+)\.attn\.rmsnorm_proj_qwb_iq_rope\.wq_b\.weight": r"layers.\1.attn.wq_b.weight",
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qwb_iq_rope\.wq_b\.scale": r"layers.\1.attn.wq_b.scale",
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qwb_iq_rope\.q_norm\.weight": r"layers.\1.attn.q_norm.weight",
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qwb_iq_rope\.wq_b_iq\.weight": r"layers.\1.attn.indexer.wq_b.weight",
        r"layers\.(\d+)\.attn\.rmsnorm_proj_qwb_iq_rope\.wq_b_iq\.scale": r"layers.\1.attn.indexer.wq_b.scale",

        r"layers\.(\d+)\.attn\.proj_qwb\.wkv_b\.weight": r"layers.\1.attn.wkv_b.weight",
        r"layers\.(\d+)\.attn\.proj_qwb\.wkv_b\.scale": r"layers.\1.attn.wkv_b.scale",
        r"layers\.(\d+)\.attn\.kv_rmsnorm\.kv_norm\.weight": r"layers.\1.attn.kv_norm.weight",

        r"layers\.(\d+)\.attn\.layernorm_rope_rotate\.k_norm\.weight": r"layers.\1.attn.indexer.k_norm.weight",
        r"layers\.(\d+)\.attn\.layernorm_rope_rotate\.k_norm\.bias": r"layers.\1.attn.indexer.k_norm.bias",

        r"layers\.(\d+)\.attn\.proj_w\.weights_proj\.weight": r"layers.\1.attn.indexer.weights_proj.weight",

    }

    state_dict = {}
    for key, value in tilert_state_dict.items():
        is_found = False
        for pattern, target_pattern in key_casting_maps.items():
            match = re.match(pattern, key)
            if match:
                new_key = re.sub(pattern, target_pattern, key)
                if new_key in state_dict:
                    print(f"Key {new_key} already exists in state_dict")
                    continue
                state_dict[new_key] = value
                is_found = True
                break
        if not is_found:
            if key in state_dict:
                print(f"Key {key} already exists in state_dict")
                continue
            state_dict[key] = value
    return state_dict


@pytest.mark.parametrize("model_config", [])
def test_e2e_forward_pass(model_config):
    with open(model_config, "r", encoding="utf-8") as f_json:
        model_config = json.load(f_json)
    model_args = ModelArgs(**model_config)

    x = torch.randint(0, model_args.vocab_size, (1, 1))
    # When testing golden, gemm_impl in mlp.py should also be set to "bf16"
    tilert_model = TilertDeepSeekV32Transformer(model_args, enable_tilert=True)
    origin_model = DeepSeekV32Transformer(model_args)
    print(f"Element number of tilert_model.state_dict(): {len(tilert_model.state_dict())}")
    print(f"Element number of origin_model.state_dict(): {len(origin_model.state_dict())}")
    conv_dict = convert_state_dict(tilert_model.state_dict())
    # print the element number of the conv_dict
    print(f"Element number of conv_dict: {len(conv_dict)}")

    origin_model.load_state_dict(conv_dict)


    ref_output = origin_model(x, start_pos=127)
    tilert_output = tilert_model(x, start_pos=127)
    abs_err = torch.abs(ref_output - tilert_output)
    rel_err = abs_err / torch.abs(ref_output)
    print(f"Rel err: max-{rel_err.max():.6f}/mean-{rel_err.mean():.6f}")
    print(f"Abs err: max-{abs_err.max():.6f}/mean-{abs_err.mean():.6f}")
    print("Ref:", ref_output)
    print("Tilert:", tilert_output)
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_output.flatten(),
        tilert_output.flatten(),
        dim=0
    )
    print(f"Cosine similarity: {cos_sim.item():.6f}")



def main():
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    test_e2e_forward_pass("config_671B_v3.2_layer2_device1.json")


if __name__ == "__main__":
    main()
