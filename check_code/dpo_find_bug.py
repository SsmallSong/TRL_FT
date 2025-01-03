from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer
import torch
import json
import os
import pickle as pkl
import random
import numpy as np
from vllm import LLM, SamplingParams
messages_1 = [
    {"role": "user", "content": "What is your favourite condiment?"},
   {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
   ]

def process(row):
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row
device = "cuda" # the device to load the model onto
# export CUDA_VISIBLE_DEVICES=1
model_id_base="mistralai/Mistral-7B-Instruct-v0.2" 
model_id='/home/wxt/huatong/huggingface/hub/mistral_7b_instruct_dpo_new'
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left',legacy=False)
print("Tokenizer Loading Finished!")
model_base = AutoModelForCausalLM.from_pretrained(model_id_base).eval()
model = AutoModelForCausalLM.from_pretrained(model_id).eval()
# model = LLM(model=model_id, tensor_parallel_size=1,
#                   trust_remote_code=True)
print("Model Loading Finished!")
model.generation_config.pad_token_id = model.generation_config.eos_token_id

dataset_id='snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset'
ds = load_dataset(dataset_id)
# print(ds)

# ds = ds.map(
#     process,
# #    num_proc=multiprocessing.cpu_count(),
#     load_from_cache_file=False,
# )

train_dataset = ds["train_iteration_3"]
eval_dataset = ds["test_iteration_3"]
messages_2=train_dataset[0]['chosen']
# print("messages_2:",messages_2)

encodeds = tokenizer.apply_chat_template(messages_1, return_tensors="pt",tokenize=True)
encodeds_2 = tokenizer.apply_chat_template(messages_1, return_tensors="pt",tokenize=False)
print('enceodeds:',encodeds)
print('enceodeds_2:',encodeds_2)
# encodeds = tokenizer.apply_chat_template(messages_2, return_tensors="pt",tokenize=True)
# encodeds_2 = tokenizer.apply_chat_template(messages_2, return_tensors="pt",tokenize=False)
# print('enceodeds:',encodeds)
# print('enceodeds_2:',encodeds_2)

model_inputs = encodeds.to(device)
model.to(device)
model_base.to(device)
for i in range(3):
    print("Base-model Output:")
    # sampling_params=SamplingParams(temperature=0,max_tokens=2048,n=1)
    generated_ids_base = model_base.generate(model_inputs, max_new_tokens=1024, do_sample=True)
    decoded_base = tokenizer.batch_decode(generated_ids_base)
    # decoded=gengeated_ids[0].outputs[0].text.strip()
    print("base-model-{}:{}".format(i,decoded_base[0]))
    print('==================================================')

for j in range(3):
    print("New-model Output:")
    generated_ids = model.generate(model_inputs, max_new_tokens=1024, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    print("new-model-{}:{}".format(j,decoded[0]))
    print('==================================================')
# model_inputs = encodeds_2.to(device)
#sampling_params=SamplingParams(temperature=0,max_tokens=2048,n=1)
#generated_ids = model.generate(model_inputs, sampling_params)
#decoded = tokenizer.batch_decode(generated_ids)
#print("tokenize=False:",decoded[0])


