import datasets
import json
import os
import numpy as np
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from vllm import LLM, SamplingParams
import pickle as pkl

f = '/home/wxt/huatong/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl'
x = open(f).readlines()
sampling_params = SamplingParams(temperature=0, max_tokens=2048, n=1)

import argparse
def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model_id', type=str)
    args = parser.parse_args()
    return args

args = get_args()

model_id = ""
ref_model_id = ""
ref_model_id = '/mnt/data_large/ccy/hermes_mft2_ray_rl0_0.2_7_not_nor2_lora_checkpoint_6000'
model_id = '/mnt/data_large/ccy/hermes_mft2_ray_rl0_0.2_7_not_nor2_lora_checkpoint_6000'
model_id = '/mnt/data_large/ccy/hermes_mft2_ray_rl1_0.2_7_not_nor2_lora_hh_checkpoint_6500'
model_id = '/mnt/data_large/ccy/hermes_mft2_ray_rl0.5_0.2_15_not_nor2_lora_eurus_checkpoint_1500'
model_id = '/mnt/data_large/ccy/hermes_mft4/checkpoint-3000'
model_id = '/mnt/data_large/ccy/hermes_mft2_ray_rl0.5_0.25_15_not_nor2_lora_eurus_5e-5_checkpoint_2000'
model_id = '/mnt/data_large/ccy/hermes_sft_tool_base_2/checkpoint-16500'
model_id = '/mnt/data_large/ccy/mistral_rft/checkpoint-450'
model_id = '/mnt/data_large/ccy/mistral_rft_5e-6_64_capy_rl_w0.05_hh_128/checkpoint-236'
model_id = '/mnt/data_large/ccy/mistral_rft_5e-6_64_capy_rl_w0.25_hh_63_len512/checkpoint-474'
model_id = '/mnt/data_large/ccy/mistral_rft_5e-6_64_capy_rl_w0_hh_63_len512/checkpoint-177'
model_id = '/mnt/data_large/ccy/mistral_rft_5e-6_64_capy_warm33_cont/checkpoint-1180'
model_id = '/mnt/data_large/ccy/llama2_score_0.6'
model_id = '/mnt/data_large/ccy/Meta-Llama-3-8B-Instruct'
model_id = '/mnt/data_large/ccy/Llama-3-Instruct-8B-DPO'
model_id = args.model_id
mistral_temp = False
capy_temp = False
hh_temp = False
alpaca_temp = False
llama3_temp = True
res = []
tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id)

def gen_templated_instruction(prefix, suffix=''):
    template = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{response}'''
    return template.format(instruction=prefix, response=suffix)
    
if capy_temp:
    tokenizer = AutoTokenizer.from_pretrained("/mnt/data_large/ccy/mistral-orpo-capybara-7k")

'''
if not os.path.exists('mess_{}'.format(model_id.replace('/',''))):
    llm = LLM(model=model_id, tensor_parallel_size=1,
                  trust_remote_code=True)
    pos = model_id.rfind('/')
    name = model_id[pos:]
    if name.find('checkpoint') == 1:
        name = model_id[model_id.rfind('/', 0, pos):].replace('/', '')
    messes = []
    for e in x[:]:
        e = json.loads(e)
        turns = e['turns']
        question_id = e['question_id']
        mess = []
        resp_turns = []
        for t in turns:
            mess.append({'role': 'user', 'content': t})
            if alpaca_temp:
                if t == 0:
                    prompts = gen_templated_instruction(mess[0]['content'])
                else:
                    prompts = gen_templated_instruction(str(mess))
            else:
                prompts = tokenizer.apply_chat_template(mess, tokenize=False, add_generation_prompt=True)
            # if not mistral_temp and not capy_temp:
            #     prompts += "<|im_start|>assistant\n"
            # print(prompts)
            # ___ = input()

            predict = llm.generate(prompts, sampling_params)
            mess.append({'role': 'assistant', 'content': predict[0].outputs[0].text.strip()})
            resp_turns.append(predict[0].outputs[0].text.strip())
            # prefix = "<|im_start|>assistant\n{}<|im_end|>\n".format(t)
        messes.append(mess)
        res.append({'question_id': question_id,  'answer_id': question_id, 'model_id': name,
                    'choices': [{'index': 0, 'turns': resp_turns}]})
        res[-1] = json.dumps(res[-1])

    out_f = '../llama/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/{}.jsonl'.format(name)
    out_f = open(out_f, 'w')
    out_f.write('\n'.join(res))

    pkl.dump(messes, open('mess_{}'.format(model_id.replace('/', '')),'wb'))

    del llm
    torch.cuda.empty_cache()

else:
    prompts = []
    messes = pkl.load(open('mess_{}'.format(model_id.replace('/', '')), 'rb'))
    for i, e in enumerate(x[:]):
        e = json.loads(e)
        turns = e['turns']
        question_id = e['question_id']
        # mess = []
        resp_turns = []
        for j, t in enumerate(turns):
            mess = messes[i][:j*2+1]
            prompt = tokenizer.apply_chat_template(mess, tokenize=False)
            prompt += "<|im_start|>assistant\n"
            prompts.append(prompt)


new_messes = []
for e in messes:
    new_messes.append(e)
    new_messes.append(e[:2])
messes = new_messes

'''



if not os.path.exists('alpaca_{}.json'.format(model_id.replace('/', ''))):
    llm = LLM(model=model_id, tensor_parallel_size=1,
                      trust_remote_code=True)
    import datasets
    eval_set = datasets.load_dataset("mt-bench/alpaca_eval", "alpaca_eval")["eval"]
    alpaca_prompts = []
    for example in eval_set:
        if hh_temp:
            alpaca_prompts.append('\n\nHuman: {}\n\nAssistant:'.format(example["instruction"]))
        elif mistral_temp:
            alpaca_prompts.append('[INST] {} [\INST]'.format(example["instruction"]))
        elif capy_temp:
            alpaca_prompts.append('<|user|>\n{}</s>\n<|assistant|>\n'.format(example["instruction"]))
        elif alpaca_temp:
            alpaca_prompts.append(gen_templated_instruction(example["instruction"]))
        elif llama3_temp:
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': example['instruction']}], tokenize=False, add_generation_prompt=True)
            alpaca_prompts.append(prompt)
        else:
            alpaca_prompts.append('<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'.format(example["instruction"]))


    res = []
    alpaca_predicts = llm.generate(alpaca_prompts, sampling_params)
    for i, example in enumerate(eval_set):
        example["output"] = alpaca_predicts[i].outputs[0].text.strip()
        example['output'] = example['output'].replace('<|im_start|>','')
        res.append(example)
    alpaca_out = open('alpaca_{}.json'.format(model_id.replace('/', '')), 'w')
    json.dump(res, alpaca_out, ensure_ascii=False, indent=4)
    # alpaca_out.write('\n'.join(res))
    del llm
    torch.cuda.empty_cache()



messes = []
with open('alpaca_{}.json'.format(model_id.replace('/', '')), 'r', encoding='utf-8') as file:
    res = json.load(file)

for e in res:
    messes.append([{'role': 'user', 'content': e['instruction']}, {'role': 'assistant', 'content': e['output']}])

print(len(messes))
print(messes[0])


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
ray_tokenizer = transformers.AutoTokenizer.from_pretrained(
    'Ray2333/reward-model-Mistral-7B-instruct-Unified-Feedback')

batch_size=4
scores = []
for i in range(0, len(messes), batch_size):
    batch_mess = messes[i:i + batch_size]
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
    batch_mess = messes[i:i + batch_size]
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


'''
'''
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

del eu_reward_model
torch.cuda.empty_cache()
'''
'''
'''
source_texts = prompts
target_texts = [e[-1]['content'] for e in messes]
model1 = transformers.AutoModelForCausalLM.from_pretrained(
    model_id
)
model1.eval()
model1.cuda()
def batch_target_probability(model, tokenizer, source_texts, target_texts):
    # 对“source”文本进行编码
    # tokenizer.padding_side = "left"
    #tokenizer.truncation_side = "left"
    source_encodings = tokenizer(source_texts, return_tensors="pt", padding=True, truncation=True)

    # 对“target”文本进行编码，并计算概率
    # tokenizer.padding_side = "right"
    target_encodings = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True,
                                 add_special_tokens=False)
    source_encodings = {k: v.to(model.device) for k, v in source_encodings.items()}
    target_encodings = {k: v.to(model.device) for k, v in target_encodings.items()}

    with torch.no_grad():
        inputs = {key: torch.cat([source_encodings[key], target_encodings[key]], dim=-1) for key in source_encodings}
        labels = torch.cat([torch.full_like(source_encodings['input_ids'], -100), target_encodings['input_ids']],
                           dim=-1)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return torch.exp(-loss)  # 计算“target”部分的概率

probs1 = []
for s, t in zip(source_texts, target_texts):
    probs1.append(batch_target_probability(model1, tokenizer, s, t))



del model1
torch.cuda.empty_cache()
# ref_model_id = ''
model2 = transformers.AutoModelForCausalLM.from_pretrained(
    ref_model_id
)
model2.eval()
model2.cuda()
probs2 = []
for s, t in zip(source_texts, target_texts):
    probs2.append(batch_target_probability(model2, tokenizer, s, t))
# 计算KL散度（向量化操作）

probs2 = torch.tensor(probs2)
probs1= torch.tensor(probs1)


epsilon = 1e-10
probs1 += epsilon
probs2 += epsilon
kl_divergence = torch.sum(probs1 * torch.log(probs1 / probs2))
print("总KL散度为:", kl_divergence.item())
'''
#probs = batch_target_probability(model, tokenizer, source_texts, target_texts)
