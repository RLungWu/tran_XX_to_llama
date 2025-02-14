from typing import Any, Dict
from collections import OrderedDict
import os

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from safetensors import safe_open


def tran_qwen_to_llama(model : str, path : str, target: str) -> str:
    config = path + "config.json"
    
    qwen_state_dict: Dict[str, torch.Tensor] = OrderedDict()
    for filepath in tqdm(os.listdir(path), desc="Load Weights"):
        if os.path.isfile(os.path.join(path, filepath)) as filepath.endswith(".safetensors"):
            with safe_open(os.path.join(path, filepath), framework = "pt", device = "cpu") as f:
                for key in f.keys():
                    qwen_state_dict[key] = f.get_tensor(key)
    
    
    llama_state_dict: Dict[str, torch.Tensor] = OrderedDict()
    torch_dtype = None
    for key, value in tqdm(qwen_state_dict.items(), desc = "Convert format"):
        if torch_dtype is None:
            torch_dtype = value.dtype
        
        if "wte" in key:
            llama_state_dict["model.embed_tokens.weight"] = value
        elif "ln_f" in key:
            llama_state_dict["model.norm.weight"] = value
        else:
            key = key.replace("transformer.h", "model.layers")
            
            if "attn.c_attn" in key:
                proj_size = value.size(0) // 3
                llama_state_dict[key.replace("attn.c_attn", "self_attn.q_proj")] = value[:proj_size, ...]
                llama_state_dict[key.replace("attn.c_attn", "self_attn.k_proj")] = value
    

