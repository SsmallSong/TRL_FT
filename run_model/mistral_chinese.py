import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model_path = "itpossible/Chinese-Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)

text = "请为我推荐中国三座比较著名的山"
messages = [{"role": "user", "content": text}]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=300, do_sample=True)
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(outputs)
