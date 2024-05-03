# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer
import torch
device = "cuda" # the device to load the model onto
# export CUDA_VISIBLE_DEVICES=1

tokenizer = AutoTokenizer.from_pretrained("/data2/huatong/model/Mistral-7B-Instruct-v0.2", padding_side='left')
print("Tokenizer Loading Finished!")
model = AutoModelForCausalLM.from_pretrained("/data2/huatong/model/Mistral-7B-Instruct-v0.2")
print("Model Loading Finished!")


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
    {"role": "user", "content": "Do you have mayonnaise recipes?"},
    {"role": "assistant", "content": "Yes, please let me tell you."}
]


# meg_change = tokenizer.apply_chat_template(messages, return_tensors="pt",tokenize=False)
encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt",tokenize=True)

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
