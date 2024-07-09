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
        #print(messages)
        mes = tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,return_tensors="pt")
        #print(mes)
        #kill
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

messes = []
with open('alpaca_{}.json'.format('8b_llama3_ppo_openrlhf'), 'r', encoding='utf-8') as file:
    res = json.load(file)

for e in res:
    messes.append([{'role': 'user', 'content': e['instruction']}, {'role': 'assistant', 'content': e['output']}])

print(len(messes))
# print(messes[0])
print('333333333333')

ray_reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(
                'Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback',
                num_labels=1,
                torch_dtype=torch.bfloat16,
                # load_in_4bit=True,
                # bnb_4bit_compute_dtype=torch.bfloat16
            )
ray_reward_model.cuda()
ray_reward_model.eval()
ray_tokenizer = transformers.AutoTokenizer.from_pretrained('Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback')
ray_tokenizer.truncation_side = "left"
#ray_tokenizer = transformers.AutoTokenizer.from_pretrained(
 #   'Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback')

batch_size=4
scores = []
for i in range(0, len(messes), batch_size):
    batch_mess = messes[i:i+batch_size]
    # batch_tgt = predicts[i:i + batch_size]
    def get_mess(mess):
        mess = [ray_tokenizer.apply_chat_template(e, tokenize=False) for e in mess]
        # mess = self.ray_tokenizer.encode_plus(mess, padding=True, max_length=4096, truncation=True, return_tensors="pt")
        mess = ray_tokenizer(mess, padding=True, max_length=4096, truncation=True,
                                  return_tensors="pt")
        mess = {k: v.to(ray_reward_model.device) for k, v in mess.items()}

        return mess

    inputs = get_mess(batch_mess)
    with torch.no_grad():
        output = ray_reward_model(**inputs)

        # batch_score = torch.sigmoid(output.logits.float().view(-1)).detach().cpu().numpy()
        batch_score = output.logits.float().view(-1).detach().cpu().numpy()
    scores.extend(batch_score)

print('4444444444444444')


f = '/home/wxt/huatong/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl'
x = open(f).readlines()
scores_dict = {}
for i, e in enumerate(x[:]):
    e = json.loads(e)
    cate = e['category']
    if cate not in scores_dict:
        scores_dict[cate] = []
    scores_dict[cate].append(scores[i])

for k, v in scores_dict.items():
    scores_dict[k] = np.mean(scores_dict[k])

print(scores_dict)
print(np.mean(scores))

del ray_reward_model
torch.cuda.empty_cache()

print('5555555555555')
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import model_training.models.reward_model

model_name = 'OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5'
hh_tokenizer = AutoTokenizer.from_pretrained(model_name)
#self.hh_reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
hh_reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16,)
hh_reward_model.cuda()
hh_reward_model.eval()
hh_tokenizer.truncation_side = "left"



batch_size=4
scores = []
for i in range(0, len(messes), batch_size):
    batch_mess = messes[i:i+batch_size]
    # batch_tgt = predicts[i:i + batch_size]
    def get_mess(messes):
        str_res = []
        for mess in messes:
            res = ""
            for e in mess:
                if e['role'] == 'assistant':
                    res += '<|assistant|>{}{}'.format(e['content'], hh_tokenizer.eos_token)
                else:
                    res += '<|prompter|>{}{}'.format(e['content'], hh_tokenizer.eos_token)
            str_res.append(res)
        mess = hh_tokenizer(str_res, padding=True, max_length=2048, truncation=True,
                                 return_tensors="pt")
        mess = {k: v.to(hh_reward_model.device) for k, v in mess.items()}
        return mess

    inputs = get_mess(batch_mess)
    with torch.no_grad():
        output = hh_reward_model(**inputs)
        # batch_score = torch.sigmoid(output.logits.float().view(-1)).detach().cpu().numpy()
        batch_score = output.logits.float().view(-1).detach().cpu().numpy()
    scores.extend(batch_score)
print('666666666666')
f = '/home/wxt/huatong/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl'
x = open(f).readlines()
scores_dict = {}
for i, e in enumerate(x[:]):
    e = json.loads(e)
    cate = e['category']
    if cate not in scores_dict:
        scores_dict[cate] = []
    scores_dict[cate].append(scores[i])

for k, v in scores_dict.items():
    scores_dict[k] = np.mean(scores_dict[k])

print(scores_dict)
print(np.mean(scores))
del hh_reward_model
torch.cuda.empty_cache()
print('777777777777')

eu_reward_model = transformers.AutoModel.from_pretrained(
                'openbmb/Eurus-RM-7b',
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
eu_reward_model.cuda()
eu_reward_model.eval()
eu_tokenizer = transformers.AutoTokenizer.from_pretrained(
                'openbmb/Eurus-RM-7b')
eu_tokenizer.truncation_side = "left"

batch_size=1
scores = []
for i in range(0, len(messes), batch_size):
    batch_mess = messes[i:i+batch_size]
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
                    str_res += ' [\INST] ' + e['content']
            final_mess.append(str_res)
        mess = eu_tokenizer(final_mess, max_length=4096, truncation=True,
                                 return_tensors="pt")
        mess = {k: v.to(eu_reward_model.device) for k, v in mess.items()}
        return mess

    inputs = get_mess(batch_mess)
    with torch.no_grad():
        output = eu_reward_model(**inputs)
        # batch_score = torch.sigmoid(output.float().view(-1)).detach().cpu().numpy()
        batch_score = output.float().view(-1).detach().cpu().numpy()
    scores.extend(batch_score)
print(np.mean(scores))
print('888888888888')
del eu_reward_model
torch.cuda.empty_cache()

