import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from vllm import LLM, SamplingParams
import json
import numpy as np
from tqdm import tqdm
fs = []
# f = '/mnt/data_local/tmp/llama/alpaca_mntdata_largeccyhermes_mft5_rl0_eurus_1e-6_checkpoint-2000.json'
# fs.append(f)
# # 2294.0789
# # 0.14317279
#   17.18     10.83   
# f = '/mnt/data_local/tmp/llama/alpaca_mntdata_largeccyhermes_1e-6_64_mbeta0.15_hh_len2048_tgt1024_win256checkpoint-400.json'
# fs.append(f)
# # 2297
# # 0.14405085
# 17.79 11.25
# f = '/mnt/data_local/tmp/llama/alpaca_mntdata_largeccyhermes_1e-6_64_mbeta0.25_hh_len2048_tgt1024_win256checkpoint-400.json'
# fs.append(f)
# # 2276.7424
# # 0.1438
#  16.64     10.28  
# f = '/mnt/data_local/tmp/llama/alpaca_mntdata_largeccyhermes_1e-6_64_mbeta0.15_hh_len2048_tgt1024_win256checkpoint-300.json'
# fs.append(f)
# 2253.107
# 0.1435144
# 17.14     10.50 
# f = '/mnt/data_local/tmp/llama/alpaca_mntdata_largeccyhermes_1e-6_64_w0.2_hh_len2048_tgt1024_win256checkpoint-700.json'
# fs.append(f)
# 2297.0728
# 0.14419
#  16.20     10.38  
# f = '/mnt/data_local/tmp/llama/alpaca_mntdata_largeccyhermes_1e-6_64_mbeta0.15_hh_len2048_tgt1024_win256checkpoint-500.json'
# fs.append(f)
# 2307.7683
# 0.14428827
# 17.33 10.64
# f = '/mnt/data_local/tmp/llama/alpaca_mntdata_largeccyhermes_1e-6_64_mbeta0.25_hh_len2048_tgt1024_win256checkpoint-500.json'
# fs.append(f)
# 2281
# 0.14402968
#  17.42     10.55    

# f = '/mnt/data_local/tmp/llama/alpaca_mntdata_largeccyhermes_1e-6_64_mbeta0.15_hh_len2048_tgt1024_win256checkpoint-1400.json'
# fs.append(f)
# 2201.8696
# 0.1402
# f = '/mnt/data_local/tmp/llama/alpaca_mntdata_largeccyhermes_1e-6_64_meanmbeta0.25_hh_len2048_tgt1024_win256checkpoint-1400.json'
# fs.append(f)
# 2276.9734
# 0.14378971


# f = '/mnt/data_local/tmp/llama/alpaca_mntdata_largeccyhermes_1e-6_64_mbeta0.15_hh_len2048_tgt1024_win256checkpoint-700.json'
# fs.append(f)
# # 2252.9375

# 1295.2771
# 
# 3391.6226
# 0.16634998
# f = '/mnt/data_local/tmp/llama/alpaca_mntdata_largeccyllama3_5e-7_64_mbeta0.15_hh_len2048_tgt1024_win256checkpoint-700.json'
# f = '/mnt/data_large/ccy/alpaca_mntdata_largeccyLlama-3-Instruct-8B-DPO.json'

import argparse
def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--force', type=bool, default=False)
    args = parser.parse_args()
    return args

args = get_args()

f = '/home/wxt/huatong/TRL_FT/test_code/model_another_test/alpaca_{}.json'.format(args.model_id.replace('/',''))


fs.append(f)
# 1e-5
# 3091
# 0.1576191
 # 3e-6
 # 31
 # 0.15759987

 # 3038.6191
 # 0.1569

# /mnt/data_local/tmp/llama/alpaca_mntdata_largeccyllama3_1e-6_mft_e3_len2048_tgt1024checkpoint-2000.json 
 # 3100.9163
 # 

def count_eurus_reward(messes):
    eu_reward_model = transformers.AutoModel.from_pretrained(
                'openbmb/Eurus-RM-7b',
                trust_remote_code=True
            )
    eu_reward_model.cuda()
    eu_reward_model.eval()
    eu_tokenizer = transformers.AutoTokenizer.from_pretrained(
                    'openbmb/Eurus-RM-7b')
    eu_tokenizer.truncation_side = "left"
    batch_size=1
    scores = []
    for i in tqdm(range(0, len(messes), batch_size)):
        batch_mess = messes[i:i + batch_size]
        # batch_tgt = predicts[i:i + batch_size]
        def get_mess(messes):
            final_mess = []
            for m in messes:
                str_res = ""
                for e in m:
                    if e['role'] == 'system':
                        str_res += '[INST] ' + e['content']
                    if e['role'] == 'user':
                        str_res += '[INST] ' + e['content']
                    else:
                        str_res += ' [/INST] ' + e['content']
                final_mess.append(str_res)
            mess = eu_tokenizer(final_mess, max_length=4096, truncation=True,
                                    return_tensors="pt")
            mess = {k: v.to(eu_reward_model.device) for k, v in mess.items()}
            return mess

        inputs = get_mess(batch_mess)
        with torch.no_grad():
            output = eu_reward_model(**inputs)
            # batch_score = torch.sigmoid(output.float().view(-1)).detach().cpu().numpy()
            batch_score = output.float().view(-1).detach().cpu().numpy().tolist()
        scores.extend(batch_score)
    print(np.mean(scores))
    del eu_reward_model
    torch.cuda.empty_cache()
    return scores

def count_llama3_reward(messes):

    path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
            
    model = transformers.AutoModelForSequenceClassification.from_pretrained(path, device_map='cuda', 
                        trust_remote_code=True)

    eu_reward_model = model
    eu_reward_model.eval()
    eu_tokenizer = transformers.AutoTokenizer.from_pretrained(
                    path)
    eu_tokenizer.truncation_side = "left"
    batch_size=1
    scores = []
    for i in tqdm(range(0, len(messes), batch_size)):
        batch_mess = messes[i:i + batch_size]
        # batch_tgt = predicts[i:i + batch_size]
        def get_mess(messes):
            final_mess = []
            for m in messes:
                str_res = ""
                for e in m:
                    if e['role'] == 'system':
                        str_res += '<|start_header_id|>user<|end_header_id|>\n\n' + e['content'] + '<|eot_id|>'
                    if e['role'] == 'user':
                        str_res += '<|start_header_id|>user<|end_header_id|>\n\n' + e['content'] + '<|eot_id|>'
                    else:
                        str_res += '<|start_header_id|>assistant<|end_header_id|>\n\n' + e['content'] + '<|eot_id|>'
                final_mess.append(str_res)
            mess = eu_tokenizer(final_mess, max_length=4096, truncation=True,
                                    return_tensors="pt")
            mess = {k: v.to(eu_reward_model.device) for k, v in mess.items()}
            return mess

        inputs = get_mess(batch_mess)
        with torch.no_grad():
            output = eu_reward_model(**inputs)
            multi_obj_rewards = output.rewards.cpu().float() 
            # The gating layer's output is conditioned on the prompt
            gating_output = output.gating_output.cpu().float()
            # The preference score for the response, aggregated from the 
            # multi-objective rewards with the gating layer
            preference_score = output.score.cpu().float().numpy().tolist()
            batch_score = preference_score
        scores.extend(batch_score)
    print(np.mean(scores))
    del eu_reward_model
    torch.cuda.empty_cache()
    return scores


def count_hh_reward(messes):
    import model_training.models.reward_model
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_name = 'OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5'
    # model_name = '/mnt/data_large/ccy/oasst-rm-2.1-pythia-1.4b-epoch-2.5'


    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, device_map='cuda', 
                        trust_remote_code=True)

    eu_reward_model = model
    eu_reward_model.eval()
    eu_tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_name)
    eu_tokenizer.truncation_side = "left"
    batch_size=1
    scores = []
    for i in tqdm(range(0, len(messes), batch_size)):
        batch_mess = messes[i:i + batch_size]
        # batch_tgt = predicts[i:i + batch_size]
        def get_mess(messes):
            final_mess = []
            for m in messes:
                str_res = ""
                for e in m:
                    if e['role'] == 'assistant':
                        str_res += '<|assistant|>{}{}'.format(e['content'], eu_tokenizer.eos_token)
                    else:
                        str_res += '<|prompter|>{}{}'.format(e['content'], eu_tokenizer.eos_token)
                final_mess.append(str_res)
            mess = eu_tokenizer(final_mess, max_length=4096, truncation=True,
                                    return_tensors="pt")
            mess = {k: v.to(eu_reward_model.device) for k, v in mess.items()}
            return mess

        inputs = get_mess(batch_mess)
        with torch.no_grad():
            output = eu_reward_model(**inputs)
            batch_score = output.logits.float().view(-1).detach().cpu().numpy().tolist()
        scores.extend(batch_score)
    print(np.mean(scores))
    del eu_reward_model
    torch.cuda.empty_cache()
    return scores

for f in fs:
    x = json.load(open(f, 'r'))
    messes = []
    for e in tqdm(x):
        messes.append([{'role': 'user', 'content': e['instruction']}, {'role': 'assistant', 'content': e['output'].replace('<|eot_id|>', '')}])
    
    print(args.model_id)
    # if 'eurus' not in x[0]:
    #     scores = count_eurus_reward(messes)
    #     for i, e in enumerate(x):
    #         x[i]['eurus'] = scores[i]
    # else:
    #     print(np.mean([e['eurus'] for e in x]))
    if 'llama3' not in x[0]:
        scores = count_llama3_reward(messes)
        for i, e in enumerate(x):
            x[i]['llama3'] = scores[i]
    else:
        print(np.mean([e['llama3'] for e in x]))
    if 'eurus' not in x[0] or args.force:
        scores = count_eurus_reward(messes)
        for i, e in enumerate(x):
            x[i]['eurus'] = scores[i]
        json.dump(x, open(f, 'w'))
    else:
        print(np.mean([e['eurus'] for e in x]))

    if 'hh' not in x[0] or args.force:
        scores = count_hh_reward(messes)
        for i, e in enumerate(x):
            x[i]['hh'] = scores[i]
        json.dump(x, open(f, 'w'))
    else:
        print(np.mean([e['hh'] for e in x]))
    
    #count_llama3_reward(messes)
    

    
