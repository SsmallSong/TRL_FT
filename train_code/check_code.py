#import some packages and reward funcs
import model_training.models.reward_model
import os
import argparse
import json
import tqdm
# import torch
from datetime import timedelta
import torch.nn.functional as F
from typing import Dict, Union, Type, List
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
def elete_dict(d: Dict):
    """Delete all items inside the dict."""
    for k in list(d.keys()):
        del d[k]
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import torch
print(torch.cuda.device_count())


#kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
#accelerator = Accelerator(kwargs_handlers=[kwargs])# **accelerator_log_kwargs)
#rank = int(os.environ['RANK'])
#rank_sum = accelerator.num_processes
#model_device = "cuda:{}".format(rank)

model_name_or_path ='daryl149/llama-2-7b-hf'
model_config = AutoConfig.from_pretrained(model_name_or_path )
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,config=model_config)#.to(model_device)

print("begin load model")
ckpt_path = f"/home/wxt/.cache/huggingface/hub/llama2_7b_sft_halos_2_3/LATEST/policy.pt"
state_dict = torch.load(ckpt_path, map_location='cpu')
model.load_state_dict(state_dict['state'])
delete_dict(state_dict)
gc.collect()
torch.cuda.empty_cache()
print('loaded pre-trained weights')
