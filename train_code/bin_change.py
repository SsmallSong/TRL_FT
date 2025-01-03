# import torch
# bin_file_path = '/home/wxt/huggingface/hub/llama2_sft_mirror/pytorch_model-00001-of-00002.bin'  # 请将此路径替换为你的bin文件路径
# binary_data_1 = torch.load(bin_file_path, map_location='cpu')
# bin_file_path = '/home/wxt/huggingface/hub/llama2_sft_mirror/pytorch_model-00002-of-00002.bin'  # 请将此路径替换为你的bin文件路径
# binary_data_2 = torch.load(bin_file_path, map_location='cpu')

# print(len(binary_data_1)+len(binary_data_2))
# # print(binary_data_1)
# set1 = set(binary_data_1.keys())
# set2 = set(binary_data_2.keys())
# bin_set = set1.union(set2)

# print("="*60)
# print("="*60)
# print("="*60)


# policy =torch.load('/home/wxt/huatong/huggingface/hub/llama2_7b_sft_halos_2_3/LATEST/policy.pt')
# print(len(policy))
# state_dict=policy['state']
# print(len(state_dict))
# # print((state_dict))
# pt_set=set(state_dict.keys())

# # 找到 set1 中有但 set2 中没有的元素
# diff_set1 = bin_set - pt_set

# # 找到 set2 中有但 set1 中没有的元素
# diff_set2 = pt_set - bin_set

# # 打印差集合
# print("Elements in bin but not in pt:", diff_set1)
# print("Elements in pt but not in bin:", diff_set2)

import torch
from collections import OrderedDict
bin_file_path_1 = '/home/wxt/huggingface/hub/llama2_sft_mirror/pytorch_model-00001-of-00002.bin'  # 请将此路径替换为你的bin文件路径
binary_data_1 = torch.load(bin_file_path_1, map_location='cpu')

bin_file_path_2 = '/home/wxt/huggingface/hub/llama2_sft_mirror/pytorch_model-00002-of-00002.bin'  # 请将此路径替换为你的bin文件路径
binary_data_2 = torch.load(bin_file_path_2, map_location='cpu')

policy =torch.load('/home/wxt/huatong/huggingface/hub/llama2_7b_sft_halos_2_3/LATEST/policy.pt')
state_dict=policy['state']
# 更新 binary_data_1 和 binary_data_2
for key, value in state_dict.items():
    if key in binary_data_1:
        binary_data_1[key] = value
    if key in binary_data_2:
        binary_data_2[key] = value

# 保存更新后的 binary_data_1 和 binary_data_2 回 bin 文件
torch.save(binary_data_1, bin_file_path_1)
torch.save(binary_data_2, bin_file_path_2)
