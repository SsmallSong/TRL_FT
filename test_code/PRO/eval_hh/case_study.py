
from transformers import (
    AutoConfig,
    AutoTokenizer,
    LlamaTokenizer,
    AutoModelForCausalLM
)
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

import torch
print(torch.cuda.device_count())

if __name__ == "__main__":

    model_name_or_path ='daryl149/llama-2-7b-hf'
    model_config = AutoConfig.from_pretrained(model_name_or_path )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,config=model_config).to(device)
    print("Finished load !")
  
