import datasets
import json
import os
import numpy as np
import random
import torch

from collections import OrderedDict
import transformers
from vllm import LLM, SamplingParams
import pickle as pkl
from transformers import AutoModelForCausalLM
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# f = '/home/wxt/huatong/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl'
# x = open(f).readlines()

sampling_params = SamplingParams(temperature=0, max_tokens=2048, n=1)

#load_dict_path_list=["llama2_7b_sft_halos_2_3/LATEST/policy.pt","llama2_7b_dpo_halos_beta01/LATEST/policy.pt","llama2_7b_kto_halos_beta01/LATEST/policy.pt","llama2_7b_ppo_halos_2/LATEST/policy.pt"]
# load_dict_path_list=["llama2_7b_ppo_halos_2/LATEST/policy.pt"]
cache_path="/home/wxt/.cache/huggingface/hub"
model_id="/home/wxt/.cache/huggingface/hub/7b_llama3_inst_ppo_openrlhf"

# for load_dict_path in load_dict_path_list:
    # now_dict=load_dict_path.split("/")[0]
    
    # bin_file_path_1 = '/home/wxt/huggingface/hub/llama2_sft_mirror/pytorch_model-00001-of-00002.bin'  # 请将此路径替换为你的bin文件路径
    # binary_data_1 = torch.load(bin_file_path_1, map_location='cpu')
    
    # bin_file_path_2 = '/home/wxt/huggingface/hub/llama2_sft_mirror/pytorch_model-00002-of-00002.bin'  # 请将此路径替换为你的bin文件路径
    # binary_data_2 = torch.load(bin_file_path_2, map_location='cpu')
    
    # state_dict_all = torch.load(os.path.join(cache_path, load_dict_path), map_location='cpu')
    # state_dict=state_dict_all['state']
    
    # # 更新 binary_data_1 和 binary_data_2
    # for key, value in state_dict.items():
    #     if key in binary_data_1:
    #         binary_data_1[key] = value
    #     if key in binary_data_2:
    #         binary_data_2[key] = value
    
    # # 保存更新后的 binary_data_1 和 binary_data_2 回 bin 文件
    # torch.save(binary_data_1, bin_file_path_1)
    # torch.save(binary_data_2, bin_file_path_2)
    
mistral_temp = False
res = []
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id,legacy=False)
print('1111111111111')


# if not os.path.exists('alpaca_{}.json'.format(now_dict.replace('/', ''))):

if not os.path.exists('alpaca_{}.json'.format('8b_llama3_ppo_openrlhf')):
    llm = LLM(model=model_id, tensor_parallel_size=1,trust_remote_code=True)
    
    import datasets
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval",trust_remote_code=True)["eval"]
    alpaca_prompts = []

    
    for example in eval_set:
        # alpaca_prompts.append('\n<|user|>\n{}\n<|assistant|>\n'.format(example["instruction"]))
        messages = [
            {"role": "user", "content": example["instruction"]},
        ]
      #  print(messages)
        
        mes = tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,return_tensors="pt")
      #  print(mes)
        #q_mes = tokenizer.apply_chat_template(messages,add_generation_prompt=False,tokenize=False,return_tensors="pt")
       # print(q_mes)
        alpaca_prompts.append(mes)
       # print(alpaca_prompts)
       # kill
        
        
    res = []
    alpaca_predicts = llm.generate(alpaca_prompts, sampling_params)
    for i, example in enumerate(eval_set):
        example["output"] = alpaca_predicts[i].outputs[0].text.strip()
        example['output'] = example['output'].replace('<|im_start|>','')
        res.append(example)
    alpaca_out = open('alpaca_{}.json'.format('8b_llama3_ppo_openrlhf'), 'w')
    json.dump(res, alpaca_out, ensure_ascii=False, indent=4)
    del llm
    torch.cuda.empty_cache()
    
