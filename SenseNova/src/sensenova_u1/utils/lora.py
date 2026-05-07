import torch
import torch.nn as nn
from safetensors.torch import safe_open
import gc
from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

def build_lora_names(key, lora_down_key, lora_up_key, is_native_weight):
    base = "diffusion_model." if is_native_weight else ""
    lora_down = base + key.replace(".weight", lora_down_key)
    lora_up = base + key.replace(".weight", lora_up_key)
    lora_alpha = base + key.replace(".weight", ".alpha")
    return lora_down, lora_up, lora_alpha


def load_and_merge_lora_weight(
    model: nn.Module,
    lora_state_dict: dict,
    lora_down_key: str = ".lora_down.weight",
    lora_up_key: str = ".lora_up.weight",
):
    is_native_weight = any("diffusion_model." in key for key in lora_state_dict)
    for key, value in model.named_parameters():
        lora_down_name, lora_up_name, lora_alpha_name = build_lora_names(
            key, lora_down_key, lora_up_key, is_native_weight
        )
        if lora_down_name in lora_state_dict:
            lora_down = lora_state_dict[lora_down_name]
            lora_up = lora_state_dict[lora_up_name]
            lora_alpha = float(lora_state_dict[lora_alpha_name])
            rank = lora_down.shape[0]
            scaling_factor = lora_alpha / rank
            assert lora_up.dtype == torch.float32
            assert lora_down.dtype == torch.float32
            delta_W = scaling_factor * torch.matmul(lora_up, lora_down).to(value.device)
            value.data = (value.data + delta_W).type_as(value.data)
    return model


def load_and_merge_lora_weight_from_safetensors(
    model: nn.Module,
    lora_weight_path: str,
    lora_down_key: str = ".lora_down.weight",
    lora_up_key: str = ".lora_up.weight",
):
    lora_state_dict = {}
    with safe_open(lora_weight_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_state_dict[key] = f.get_tensor(key)
    model = load_and_merge_lora_weight(model, lora_state_dict, lora_down_key, lora_up_key)
    return model




def apply_loras_gguf(
    model_sd,
    lora_sd,
):
    sd = {}
    for key, weight in model_sd.items():
        if weight is None:
            continue
        device = weight.device
        deltas_dtype =  torch.bfloat16
        deltas = _prepare_deltas(lora_sd, key, deltas_dtype, device)
        if deltas is None:
            sd[key] = weight
        else:
            deltas = deltas.to(dtype=deltas_dtype)
            if  getattr(weight,"quant_type",False):
                try:
                    weight = (dequantize_gguf_tensor(weight).to(dtype=deltas_dtype)) + deltas
                    sd[key] = weight
                except Exception as e:
                    print(f"Error dequantizing GGUF weight for {key}: {e}")
                    sd[key] = weight
            else:
                sd[key] = weight + deltas
            
        del weight,deltas
    del model_sd
    gc.collect()
    return sd

def _prepare_deltas( lora_sd,key: str, dtype: torch.dtype, device: torch.device
) -> torch.Tensor | None:
    deltas = None
    prefix = key[: -len(".weight")]
    key_a = f"{prefix}.lora_down.weight"
    key_b = f"{prefix}.lora_up.weight"
    lora_alpha = f"{prefix}.alpha"
    if key_a  in lora_sd :
        lora_down = lora_sd[key_a].to(device=device)
        lora_up = lora_sd[key_b].to(device=device)
        alpha = float(lora_sd.get(lora_alpha, 1.0))
        rank = lora_down.shape[0]
        scaling_factor = alpha / rank
        deltas = scaling_factor * torch.matmul(lora_up, lora_down).to(device)
        del lora_down, lora_up,alpha
    return deltas



