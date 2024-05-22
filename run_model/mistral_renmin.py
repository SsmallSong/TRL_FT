# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer
import torch
#device = "cuda" # the device to load the model onto
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# export CUDA_VISIBLE_DEVICES=1
# model_id='/home/wxt/huatong/huggingface/hub/mistral_7b_instruct'
model_id='mistralai/Mistral-7B-Instruct-v0.2'
# model_id='daryl149/llama-2-7b-hf'
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
print("Tokenizer Loading Finished!")
model = AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.bfloat16, device_map=device)
print("Model Loading Finished!")

model.generation_config.pad_token_id = model.generation_config.eos_token_id
# print( model.generation_config.eos_token_id)
# print(model.generation_config.pad_token_id)

messages_1 = [
    {"role": "user", "content": "2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？"},

   ]
messages_2 = [
    {"role": "user", "content": "2024年是中国红十字会成立多少周年?"},
   {"role": "assistant", "content": "10"},
    {"role": "user", "content": "2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？"}
   ]

messages_3 = [
            {"role": "user", "content": "On March 18, 2024, during his inspection tour in Hunan Province, General Secretary Xi Jinping made his first stop at which school?"},

               ]
model_inputs= tokenizer.apply_chat_template(messages_1, return_tensors="pt").to(device)

#model_inputs = encodeds.to(device)
#model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=200, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(decoded[0])
kill
model_inputs = tokenizer.apply_chat_template(messages_2, return_tensors="pt").to(device)

#model_inputs = encodeds.to(device)
#model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(decoded[0])
