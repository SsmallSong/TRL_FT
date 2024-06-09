import torch
from transformers import AutoConfig

# # 加载模型配置
# model_name_or_path = 'path_to_your_model_directory'  # 请将此路径替换为你的模型目录路径
# config = AutoConfig.from_pretrained(model_name_or_path)

# 加载模型权重
bin_file_path = '/home/wxt/huggingface/hub/llama2_sft_mirror/pytorch_model-00001-of-00002.bin'  # 请将此路径替换为你的bin文件路径
state_dict = torch.load(bin_file_path, map_location='cpu')

# 打印模型参数的键和值
for key, value in state_dict.items():
    print(f"Parameter: {key}, Shape: {value.shape}")

# 如果只想查看键，可以使用以下代码
print(state_dict.keys())
