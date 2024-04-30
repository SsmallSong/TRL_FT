import datasets
import json
import os
import numpy as np
import random
import torch
import transformers
from vllm import LLM, SamplingParams
import pickle as pkl

f = '/home/wxt/huatong/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl'
x = open(f).readlines()
sampling_params = SamplingParams(temperature=0, max_tokens=2048, n=1)

# ref_model_id = 'hermes_mft2_ray_rl0_0.2_7_not_nor2_lora_checkpoint_6000'
model_id = '/home/wxt/huatong/huggingface/hub/mistral_7b_instruct_dpo'

mistral_temp = False 
res = []
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id,legacy=False)

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
            prompts = tokenizer.apply_chat_template(mess, tokenize=False)
            # prompts = tokenizer.decode(inputs[0])
            # print(prompts)
            # ___ = input()
            
            prompts += "<|im_start|>assistant\n"
            predict = llm.generate(prompts, sampling_params)
            mess.append({'role': 'assistant', 'content': predict[0].outputs[0].text.strip()})
            resp_turns.append(predict[0].outputs[0].text.strip())
            # prefix = "<|im_start|>assistant\n{}<|im_end|>\n".format(t)
        messes.append(mess)
        res.append({'question_id': question_id,  'answer_id': question_id, 'model_id': name,
                    'choices': [{'index': 0, 'turns': resp_turns}]})
        res[-1] = json.dumps(res[-1])

    out_f = '/home/wxt/huatong/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/{}.jsonl'.format(name)
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

if not os.path.exists('alpaca_{}.json'.format(model_id.replace('/', ''))):
    llm = LLM(model=model_id, tensor_parallel_size=1,
                      trust_remote_code=True)
    import datasets
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval",trust_remote_code=True)["eval"]
    alpaca_prompts = []
    for example in eval_set:
        if mistral_temp:
                alpaca_prompts.append('[INST] {} [\INST]'.format(example["instruction"]))
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



ray_reward_model = transformers.AutoModelForSequenceClassification.from_pretrained(
                'reward-model-Mistral-7B-instruct-Unified-Feedback',
                num_labels=1,
                torch_dtype=torch.bfloat16,
                # load_in_4bit=True,
                # bnb_4bit_compute_dtype=torch.bfloat16
            )
ray_reward_model.cuda()
ray_reward_model.eval()
ray_tokenizer = transformers.AutoTokenizer.from_pretrained('reward-model-Mistral-7B-instruct-Unified-Feedback')
ray_tokenizer.truncation_side = "left"
ray_tokenizer = transformers.AutoTokenizer.from_pretrained(
    'reward-model-Mistral-7B-instruct-Unified-Feedback')

batch_size=4
scores = []
for i in range(0, len(messes), batch_size):
    batch_mess = messes[i*batch_size:(i+1)*batch_size]
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

model_name = 'oasst-rm-2.1-pythia-1.4b-epoch-2.5'
hh_tokenizer = AutoTokenizer.from_pretrained(model_name)
#self.hh_reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
hh_reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16,)
hh_reward_model.cuda()
hh_reward_model.eval()
hh_tokenizer.truncation_side = "left"



batch_size=4
scores = []
for i in range(0, len(messes), batch_size):
    batch_mess = messes[i*batch_size:(i+1)*batch_size]
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


eu_reward_model = transformers.AutoModel.from_pretrained(
                'Eurus-RM-7b',
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
eu_reward_model.cuda()
eu_reward_model.eval()
eu_tokenizer = transformers.AutoTokenizer.from_pretrained(
                'Eurus-RM-7b')
eu_tokenizer.truncation_side = "left"

batch_size=1
scores = []
for i in range(0, len(messes), batch_size):
    batch_mess = messes[i*batch_size:(i+1)*batch_size]
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
