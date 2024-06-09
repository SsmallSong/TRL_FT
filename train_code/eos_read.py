from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
model_name_or_path = "/home/wxt/huggingface/hub/llama2_sft_mirror/"  # 请将此处替换为具体的模型名称或路径
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# 获取 EOS token 的 ID
eos_token_id = tokenizer.stop_token_id

# 获取 EOS token 的字符串表示
eos_token = tokenizer.decode([eos_token_id])

print(f"EOS token ID: {eos_token_id}")
print(f"EOS token: {eos_token}")
