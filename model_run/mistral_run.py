# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer
import torch

device = "cuda"  # the device to load the model onto
# export CUDA_VISIBLE_DEVICES=1

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
print("Tokenizer Loading Finished!")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
print("Model Loading Finished!")

# Set padding_side='left' for tokenizer
tokenizer.padding_side = "left"

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

# Encode messages
encodeds = tokenizer(messages, return_tensors="pt", padding=True, truncation=True)

model_inputs = encodeds.to(device)
model.to(device)

# Generate text with left-padding
generated_ids = model.generate(model_inputs, max_length=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(decoded[0])
