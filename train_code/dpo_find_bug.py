from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer
import torch
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
model_id="mistralai/Mistral-7B-Instruct-v0.2" 
model_id='/home/wxt/huatong/huggingface/hub/mistral_7b_instruct_dpo'
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
print("Tokenizer Loading Finished!")
model = AutoModelForCausalLM.from_pretrained(model_id).eval()
print("Model Loading Finished!")
model.generation_config.pad_token_id = model.generation_config.eos_token_id

dataset_id='snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset'
ds = load_dataset(dataset_id)
print(ds)

ds = ds.map(
    process,
    num_proc=multiprocessing.cpu_count(),
    load_from_cache_file=False,
)

train_dataset = ds["train_iteration_3"]
eval_dataset = ds["test_iteration_3"]
messages_2=train_dataset[0]['chosen']
print("messages_2:",messages_2)

encodeds = tokenizer.apply_chat_template(messages_1, return_tensors="pt",tokenize=True)
encodeds_2 = tokenizer.apply_chat_template(messages_1, return_tensors="pt",tokenize=False)
print('enceodeds:',encodeds)
print('enceodeds_2:',encodeds_2)

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print("tokenize=True:",decoded[0])

model_inputs = encodeds_2.to(device)
generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print("tokenize=False:",decoded[0])
