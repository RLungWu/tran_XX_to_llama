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
    

