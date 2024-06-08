#import some packages and reward funcs
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
    AutoModelForCausalLM
)

import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
import torch
print(torch.cuda.device_count())

model_name_or_path ='daryl149/llama-2-7b-hf'
model_config = AutoConfig.from_pretrained(model_name_or_path )
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,config=model_config).to(model_device)

value_model = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5", num_labels=1)
reward_model = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5", num_labels=1)
    
