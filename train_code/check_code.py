#import some packages and reward funcs
import model_training.models.reward_model
import os
import argparse
import json
import tqdm
#import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)
def delete_dict(d: Dict):
    """Delete all items inside the dict."""
    for k in list(d.keys()):
        del d[k]
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import torch
print(torch.cuda.device_count())

model_name_or_path ='daryl149/llama-2-7b-hf'
model_config = AutoConfig.from_pretrained(model_name_or_path )
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,config=model_config).to(model_device)

print("load model")
ckpt_path = f"/home/wxt/.cache/huggingface/hub/{args.model_ckpt}/LATEST/policy.pt"
state_dict = torch.load(ckpt_path, map_location='cpu')
# step, metrics = state_dict['step_idx'], state_dict['metrics']
model.load_state_dict(state_dict['state'])
delete_dict(state_dict)
gc.collect()
torch.cuda.empty_cache()
print('loaded pre-trained weights')
