import torch
bin_file_path = '/home/wxt/huggingface/hub/llama2_sft_mirror/pytorch_model-00001-of-00002.bin'  # 请将此路径替换为你的bin文件路径
binary_data_1 = torch.load(bin_file_path, map_location='cpu')
bin_file_path = '/home/wxt/huggingface/hub/llama2_sft_mirror/pytorch_model-00002-of-00002.bin'  # 请将此路径替换为你的bin文件路径
binary_data_2 = torch.load(bin_file_path, map_location='cpu')
num=0
print(len(binary_data_1)+len(binary_data_2))
# for key , value in binary_data_1.items():
#     num+=1
#     if num==10:
#         break
#     print(key)
#     print(value)

print(binary_data_1)

print("="*20)
print("="*20)
print("="*20)

# for key,value in binary_data.items():
#     print(key)
#     print(value)
#     break
# kill
# 加载policy.pt文件
policy =torch.load('/home/wxt/huatong/huggingface/hub/llama2_7b_sft_halos_2_3/LATEST/policy.pt')
cont=1
print(len(policy))
state_dict=policy['state']
print(len(state_dict))
print((state_dict))
# # kill
# for key,value in state_dict.items():
#      cont+=1
# #     print("="*20)
#      print(key)
#      print(value)
# #     print("="*20)
#      if cont==10:
#          break
