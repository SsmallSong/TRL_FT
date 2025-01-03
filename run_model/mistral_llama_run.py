# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer
import torch
device = "cuda" # the device to load the model onto
# export CUDA_VISIBLE_DEVICES=1
model_id='/home/wxt/huatong/huggingface/hub/mistral_7b_instruct_dpo'
model_id='daryl149/llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
print("Tokenizer Loading Finished!")
model = AutoModelForCausalLM.from_pretrained(model_id)
print("Model Loading Finished!")

model.generation_config.pad_token_id = model.generation_config.eos_token_id
print( model.generation_config.eos_token_id)
print(model.generation_config.pad_token_id)
# #Text Generated:
# prompt_text = "Tell me a cute story"
# input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
# output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1, 
#                             pad_token_id=tokenizer.eos_token_id)
# generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print("Generated Text:")
# print(generated_text)
messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
   {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
   ]

# meg_change = tokenizer.apply_chat_template(messages, return_tensors="pt",tokenize=False)
#encodeds = tokeizer.apply_chat_template(messages, return_tensors="pt",tokenize=True)
prompt_llama="What is your favourite condiment?"
encodeds=tokenizer.encode(prompt_llama,return_tensors="pt")
model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
#decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print("Generated Text:")
print(decoded[0])

