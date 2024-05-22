import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model_path = "itpossible/Chinese-Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)

text = "请为我推荐中国三座比较著名的山"
# messages = [{"role": "user", "content": text}]
messages_1 = [
    {"role": "user", "content": "2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？"},

   ]
messages_2 = [
    {"role": "user", "content": "2024年是中国红十字会成立多少周年?"},
   {"role": "assistant", "content": "10"},
    {"role": "user", "content": "2024年3月18日,习近平总书记在湖南考察期间第一站来到了哪所学校？"}
   ]
inputs = tokenizer.apply_chat_template(messages_1, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=300, do_sample=True)
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(outputs)

inputs = tokenizer.apply_chat_template(messages_2, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=300, do_sample=True)
outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(outputs)
