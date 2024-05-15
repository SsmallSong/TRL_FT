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
# kill
def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--index', type=str)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--directory', default="best_checkpoint", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
    accelerator = Accelerator(kwargs_handlers=[kwargs])# **accelerator_log_kwargs)
    rank = int(os.environ['RANK'])
    rank_sum = accelerator.num_processes
    # model_name_or_path = os.path.join("..", "checkpoints", f"index_{args.index}", f"stage_{args.stage}", f"{args.directory}")
    model_name_or_path ='daryl149/llama-2-7b-hf'
    model_device = "cuda:{}".format(rank)

    model_config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,config=model_config).to(model_device)
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(model_device)
    # if args.stage==3:
    state_dict = torch.load('/home/wxt/.cache/huggingface/hub/llama2_7b_ppo_halos_1/LATEST/policy.pt', map_location='cpu')
    # step, metrics = state_dict['step_idx'], state_dict['metrics']
    model.load_state_dict(state_dict['state'])
    delete_dict(state_dict)
    gc.collect()
    torch.cuda.empty_cache()
    print('loaded pre-trained weights')

    # if args.stage==4:
    #     state_dict = torch.load('/home/wxt/.cache/huggingface/hub/llama2_7b_sft_halos_2_3/LATEST/policy.pt', map_location='cpu')
    #     # step, metrics = state_dict['step_idx'], state_dict['metrics']
    #     model.load_state_dict(state_dict['state'])
    #     delete_dict(state_dict)
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     print('loaded pre-trained weights')

    if accelerator.is_main_process:
        print(type(model))
        print(model.config)
    # set tokenizer
    if model.config.architectures[0].lower() == "llamaforcausallm":
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path,legacy=False)
        tokenizer.unk_token = "<unk>"
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,legacy=False)
    # print(type(tokenizer.eos_token))
    # print(tokenizer.eos_token)
    # print(tokenizer.eos_token_id)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.pad_token_id=tokenizer.eos_token_id
   # tokenizer.pad_token="</s>",
   # tokenizer.pad_token_id=2,
    tokenizer.sep_token = "<sep>"
    model.resize_token_embeddings(len(tokenizer))
      
    torch.cuda.empty_cache()
    model.eval()
    print(f"Rank {rank} is activated...")
    # print(model.device)
    if accelerator.is_main_process:
        for file_name in [
            "harmless_base.json",
            "helpful_base.json",
            "helpful_online.json",
            "helpful_rejection.json"
        ]:
            save_path = os.path.join("/home/wxt/huatong/TRL_FT/test_code/PRO/eval_hh/data_eval", "infer_generate_main_{}_{}_{}".format(args.index, args.stage, file_name))
            if os.path.exists(save_path):
                os.remove(save_path)
    accelerator.wait_for_everyone()
    
    for file_name in [
        "harmless_base.json",
        "helpful_base.json",
        "helpful_online.json",
        "helpful_rejection.json"
    ]:
        file_path = os.path.join("..", "data", "hh_test", file_name)
        with open(file_path, "r", encoding='utf-8') as f:
            infer_data = {line_index: json.loads(l) for line_index, l in enumerate(f.readlines()) if (line_index-rank) % rank_sum == 0}

        for line_index in infer_data:
            infer_data[line_index]["line_index"] = line_index
        infer_data = [infer_data[line_index] for line_index in infer_data]

        raw_prefixes = [l['prefix'][0] for l in infer_data]
        # list of raw prefixes
        prompts = []
        for prefix in raw_prefixes:
            prefix = "".join(prefix)
            prefix = prefix.replace("<|prompter|>", "\n\nHuman: ").replace("<|assistant|>", "\n\nAssistant: ").rstrip()
            prompts.append(prefix)
        
        setup_seed()
        generated_suffixes, truncated_prompts = generate_pipeline(model, tokenizer, prompts, add_special_tokens=True)
        setup_seed()        
        save_path = os.path.join("/home/wxt/huatong/TRL_FT/test_code/PRO/eval_hh/data_eval", "infer_generate_main_{}_{}_{}".format(args.index, args.stage, file_name))
        
        for index in range(len(infer_data)):
            infer_data[index]['infer'] = {"t": generated_suffixes[index]}
        with open(save_path, 'a', encoding='utf-8') as f:
            for line in infer_data:
                content = json.dumps(line, ensure_ascii=False)
                f.write(content+'\n')
        
        accelerator.wait_for_everyone()
        
        print("")
        if accelerator.is_main_process:
            print("Eval on {}".format(file_name))
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()
