
#import some packages and reward funcs
import os
import argparse
import json
import tqdm
#import torch
import torch.nn.functional as F
from typing import Dict, Union, Type, List
import metrics2
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM
)
from infer_func_now import setup_seed, generate_pipeline
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import torch
print(torch.cuda.device_count())
def delete_dict(d: Dict):
    """Delete all items inside the dict."""
    for k in list(d.keys()):
        del d[k]

if __name__ == "__main__":
    args = get_args()
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[kwargs])# **accelerator_log_kwargs)
    model_name_or_path ='daryl149/llama-2-7b-hf'
    model_device = "cuda:{}".format(rank)

    model_config = AutoConfig.from_pretrained(model_name_or_path )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,config=model_config).to(model_device)
    print("Finished load !")
  
